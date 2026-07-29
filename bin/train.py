#!/usr/bin/env python3

import logging
import os
import sys
import traceback

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import hydra
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from saicinpainting.training.trainers import make_training_model
from saicinpainting.utils import register_debug_signal_handlers, handle_ddp_subprocess, handle_ddp_parent_process, \
    handle_deterministic_config

LOGGER = logging.getLogger(__name__)


# Keys that existed on the PyTorch Lightning 1.x Trainer but were removed in 2.x.
# They are silently dropped so the (unchanged) LaMa configs keep working.
_REMOVED_TRAINER_KWARGS = (
    'log_gpu_memory', 'terminate_on_nan', 'weights_summary', 'progress_bar_refresh_rate',
    'flush_logs_every_n_steps', 'stochastic_weight_avg', 'truncated_bptt_steps',
    'reload_dataloaders_every_epoch', 'reload_dataloaders_every_n_epochs', 'automatic_optimization',
    'move_metrics_to_cpu', 'amp_backend', 'amp_level', 'track_grad_norm', 'auto_scale_batch_size',
    'auto_lr_find', 'auto_select_gpus', 'prepare_data_per_node', 'ipus', 'tpu_cores',
    'process_position', 'multiple_trainloader_mode', 'checkpoint_callback', 'progress_bar',
    'resume_from_checkpoint',  # handled separately, passed to trainer.fit(ckpt_path=...)
)


def migrate_trainer_kwargs(raw_kwargs):
    """Translate legacy (Lightning 1.x) Trainer kwargs from the LaMa configs into
    their Lightning 2.x equivalents and pick sane accelerator/strategy/devices.

    Returns (trainer_kwargs, resume_from_checkpoint).
    """
    kwargs = dict(raw_kwargs)

    resume_from_checkpoint = kwargs.get('resume_from_checkpoint', None)

    legacy_gpus = kwargs.pop('gpus', None)
    legacy_accelerator = kwargs.pop('accelerator', None)
    strategy = kwargs.pop('strategy', None)
    devices = kwargs.pop('devices', None)

    accelerator = None
    # In 1.x the ``accelerator`` field carried the distributed *strategy*.
    ddp_like = ('ddp', 'ddp_spawn', 'ddp2', 'dp', 'ddp_cpu', 'ddp_find_unused_parameters_false',
                'ddp_find_unused_parameters_true')
    if legacy_accelerator in ddp_like:
        if strategy is None:
            strategy = 'ddp' if legacy_accelerator in ('ddp', 'ddp2') else legacy_accelerator
    elif legacy_accelerator in ('gpu', 'cpu', 'auto', 'mps', 'tpu', 'cuda'):
        accelerator = 'gpu' if legacy_accelerator == 'cuda' else legacy_accelerator

    n_cuda = torch.cuda.device_count()

    # Resolve device count from the old ``gpus`` field.
    if devices is None and legacy_gpus is not None:
        if legacy_gpus in (-1, '-1'):
            devices = n_cuda if n_cuda > 0 else 1
        else:
            devices = legacy_gpus

    if accelerator is None:
        accelerator = 'gpu' if n_cuda > 0 else 'cpu'

    # Figure out the effective number of devices to decide whether DDP is needed.
    resolved_n = None
    if isinstance(devices, int):
        resolved_n = devices if devices > 0 else n_cuda
    elif isinstance(devices, (list, tuple)):
        resolved_n = len(devices)

    # Only spin up DDP when there is genuinely more than one device.
    if resolved_n is not None and resolved_n <= 1:
        strategy = 'auto'
    elif strategy is None:
        strategy = 'auto'

    if devices is None:
        devices = 'auto'

    kwargs['accelerator'] = accelerator
    kwargs['devices'] = devices
    kwargs['strategy'] = strategy

    # gradient clipping is done manually inside the LightningModule (required for
    # manual optimization), so it must not be forwarded to the Trainer.
    kwargs.pop('gradient_clip_val', None)
    kwargs.pop('gradient_clip_algorithm', None)

    # renamed key
    if 'replace_sampler_ddp' in kwargs:
        kwargs['use_distributed_sampler'] = kwargs.pop('replace_sampler_ddp')

    # drop everything that no longer exists on the 2.x Trainer
    for removed in _REMOVED_TRAINER_KWARGS:
        kwargs.pop(removed, None)

    # ``val_check_interval`` as an int must be <= the number of training batches,
    # otherwise Lightning raises. When it is >= limit_train_batches (as in the
    # stock configs) fall back to validating once at the end of each epoch.
    vci = kwargs.get('val_check_interval', None)
    ltb = kwargs.get('limit_train_batches', None)
    if isinstance(vci, int) and isinstance(ltb, int) and vci >= ltb:
        kwargs['val_check_interval'] = 1.0

    return kwargs, resume_from_checkpoint


def migrate_checkpoint_kwargs(raw_kwargs):
    """Translate legacy ModelCheckpoint kwargs to the Lightning 2.x names."""
    kwargs = dict(raw_kwargs)
    # ``period`` was deprecated in 1.4 and removed in favour of ``every_n_epochs``.
    if 'period' in kwargs:
        kwargs['every_n_epochs'] = kwargs.pop('period')
    return kwargs


@handle_ddp_subprocess()
@hydra.main(version_base=None, config_path='../configs/training', config_name='tiny_test')
def main(config: OmegaConf):
    try:
        need_set_deterministic = handle_deterministic_config(config)

        if sys.platform != 'win32':
            register_debug_signal_handlers()  # kill -10 <pid> will result in traceback dumped into log

        is_in_ddp_subprocess = handle_ddp_parent_process()

        config.visualizer.outdir = os.path.join(os.getcwd(), config.visualizer.outdir)
        if not is_in_ddp_subprocess:
            LOGGER.info(OmegaConf.to_yaml(config))
            OmegaConf.save(config, os.path.join(os.getcwd(), 'config.yaml'))

        checkpoints_dir = os.path.join(os.getcwd(), 'models')
        os.makedirs(checkpoints_dir, exist_ok=True)

        # there is no need to suppress this logger in ddp, because it handles rank on its own
        metrics_logger = TensorBoardLogger(config.location.tb_dir, name=os.path.basename(os.getcwd()))
        metrics_logger.log_hyperparams(config)

        training_model = make_training_model(config)

        trainer_kwargs, resume_from_checkpoint = migrate_trainer_kwargs(
            OmegaConf.to_container(config.trainer.kwargs, resolve=True))
        if need_set_deterministic:
            trainer_kwargs['deterministic'] = True

        checkpoint_kwargs = migrate_checkpoint_kwargs(
            OmegaConf.to_container(config.trainer.checkpoint_kwargs, resolve=True))

        trainer = Trainer(
            # there is no need to suppress checkpointing in ddp, because it handles rank on its own
            callbacks=[ModelCheckpoint(dirpath=checkpoints_dir, **checkpoint_kwargs)],
            logger=metrics_logger,
            default_root_dir=os.getcwd(),
            **trainer_kwargs
        )
        trainer.fit(training_model, ckpt_path=resume_from_checkpoint)
    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Training failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)


if __name__ == '__main__':
    main()
