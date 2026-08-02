"""saicinpainting package init.

Besides marking the package, this module performs two small, environment-level
setups that every entry point (bin/train.py, bin/predict.py, custom training
scripts, ...) needs, so they happen automatically on ``import saicinpainting``:

1. Re-register the ``env`` OmegaConf resolver used by the stock LaMa configs
   (e.g. ``resnet_pl.weights_path=${env:TORCH_HOME}``). OmegaConf >= 2.1 removed
   the built-in ``env`` resolver (it became ``oc.env``), which otherwise breaks
   config resolution with ``UnsupportedInterpolationType: env``.

2. Set the float32 matmul precision to a faster mode. On GPUs with tensor/matrix
   cores (including recent AMD/ROCm cards) this gives a substantial speed-up for
   the many conv/matmul ops with negligible quality impact. Override with the
   env var ``LAMA_MATMUL_PRECISION`` (e.g. ``highest`` for full fp32, or ``off``
   to leave the framework default untouched).
"""

import logging
import os

_LOGGER = logging.getLogger(__name__)

# --- 1. Backwards-compatible ${env:VAR} resolver ---------------------------------
try:
    from omegaconf import OmegaConf

    if not OmegaConf.has_resolver("env"):
        OmegaConf.register_new_resolver(
            "env", lambda name, default="": os.environ.get(name, default)
        )
except Exception as exc:  # pragma: no cover - never block importing the package
    _LOGGER.debug("Could not register the 'env' OmegaConf resolver: %s", exc)


# --- 2. Faster float32 matmuls ---------------------------------------------------
_matmul_precision = os.environ.get("LAMA_MATMUL_PRECISION", "high")
if _matmul_precision.lower() != "off":
    try:
        import torch

        torch.set_float32_matmul_precision(_matmul_precision)
    except Exception as exc:  # pragma: no cover
        _LOGGER.debug("Could not set float32 matmul precision: %s", exc)
