"""Helpers to run the (PyTorch Lightning 1.x era) LaMa configs on Lightning 2.x.

These translate the legacy Trainer/ModelCheckpoint kwargs found in the stock
configs into their Lightning 2.x equivalents and pick a sane
accelerator/strategy/devices combination. Kept here (rather than inline in
bin/train.py) so custom training scripts can import the exact same logic:

    from saicinpainting.training.trainers.pl_compat import (
        migrate_trainer_kwargs, migrate_checkpoint_kwargs)
"""

import logging

import torch

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
    """Translate legacy (Lightning 1.x) Trainer kwargs into their 2.x equivalents.

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
            # ``gpus: -1`` means "all GPUs". On typical AMD desktops the integrated
            # GPU is enumerated as a second, compute-incapable CUDA/HIP device, and
            # multi-GPU DDP across a dGPU+iGPU pair crashes (RCCL "invalid device
            # function"). Default to a single GPU for stability; users with several
            # real compute GPUs can opt in with trainer.kwargs.gpus=N (N>=2), and
            # pick which devices via HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES.
            devices = 1
            if n_cuda > 1:
                LOGGER.warning(
                    '%d GPUs detected but gpus=-1; using a single GPU for stability. '
                    'Set trainer.kwargs.gpus=N (N>=2) to train on multiple GPUs, and '
                    'HIP_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES to choose which ones.',
                    n_cuda)
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
