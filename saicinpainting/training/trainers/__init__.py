import logging
import torch
from saicinpainting.training.trainers.default import DefaultInpaintingTrainingModule


def get_training_model_class(kind):
    if kind == 'default':
        return DefaultInpaintingTrainingModule

    raise ValueError(f'Unknown trainer module {kind}')


def make_training_model(config):
    kind = config.training_model.kind
    kwargs = dict(config.training_model)
    kwargs.pop('kind')
    # In Lightning 2.x, distributed training is selected via ``strategy`` rather
    # than ``accelerator`` (which now denotes the device type). We keep backward
    # compatibility with the old configs that put 'ddp' into ``accelerator``.
    trainer_kwargs = config.trainer.kwargs
    strategy = str(trainer_kwargs.get('strategy', '') or '')
    legacy_accelerator = str(trainer_kwargs.get('accelerator', '') or '')
    wants_ddp = 'ddp' in strategy or legacy_accelerator == 'ddp'
    # DDP is only actually used with more than one GPU; on a single device we run
    # in plain (non-distributed) mode so the manual DistributedSampler is skipped.
    kwargs['use_ddp'] = wants_ddp and torch.cuda.device_count() > 1

    logging.info(f'Make training model {kind}')

    cls = get_training_model_class(kind)
    return cls(config, **kwargs)


def load_checkpoint(train_config, path, map_location='cuda', strict=True):
    model: torch.nn.Module = make_training_model(train_config)
    # torch >= 2.6 defaults to weights_only=True, which cannot unpickle the
    # OmegaConf hyperparameters stored inside LaMa checkpoints. These are trusted
    # local files, so we explicitly opt into the full (legacy) loader.
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state['state_dict'], strict=strict)
    model.on_load_checkpoint(state)
    return model
