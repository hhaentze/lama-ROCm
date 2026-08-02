# Running / training LaMa on AMD GPUs (ROCm)

This fork ports LaMa from its original CUDA 10.2 / Python 3.6 / PyTorch 1.8 /
PyTorch-Lightning 1.2 stack to a modern **AMD ROCm** stack so it runs and trains
on recent Radeon cards (tested target: **RX 9070 XT**, RDNA4 / `gfx1201`).

- **Python 3.10**
- **PyTorch 2.8.0 + ROCm 6.4** and **torchvision 0.23.0**
- **PyTorch Lightning 2.5.6**

These are (roughly) the lowest versions that reliably support current AMD GPUs
while keeping the LaMa code intact.

---

## 1. Prerequisites

1. A working **ROCm 6.4** installation on Linux (kernel driver `amdgpu` +
   ROCm user space). Verify with:
   ```bash
   rocminfo | grep -i gfx      # should list your GPU arch, e.g. gfx1201
   ```
2. Your user must be in the `render` and `video` groups.
3. `conda` / `mamba` (Miniforge recommended).

> **RDNA4 note:** RX 9070 XT reports as `gfx1201`. PyTorch's ROCm 6.4 wheels ship
> kernels for it. If you hit "no kernel image" / arch-not-found errors on a
> not-yet-officially-supported card, try forcing a compatible arch, e.g.:
> ```bash
> export HSA_OVERRIDE_GFX_VERSION=11.0.0
> ```
> Only use this as a fallback — it is not needed when the native `gfx1201`
> kernels load correctly.

---

## 2. Install

### Option A — conda (recommended)

```bash
conda env create -f conda_env.yml
conda activate lama
```

The environment file pulls PyTorch from the official ROCm wheel index
(`https://download.pytorch.org/whl/rocm6.4`).

### Option B — pip / venv

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install --extra-index-url https://download.pytorch.org/whl/rocm6.4 \
    torch==2.8.0+rocm6.4 torchvision==0.23.0+rocm6.4
pip install -r requirements.txt
```

### Verify the GPU is visible

```bash
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
On ROCm, `torch.cuda.*` is the correct API (HIP is exposed through the `cuda`
namespace) and `torch.version.hip` should be non-empty.

---

## 3. Running inpainting (prediction)

Nothing GPU-specific changed in the workflow. Download a pretrained model (see
the main README), then:

```bash
export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)
python bin/predict.py \
    model.path=$(pwd)/big-lama \
    indir=$(pwd)/my_dataset \
    outdir=$(pwd)/output \
    device=cuda           # 'cuda' selects the ROCm GPU (this is the default)
```

`bin/predict.py` now honours the `device` field from the prediction config
(`configs/prediction/default.yaml`), which defaults to `cuda` (the ROCm GPU) and
falls back to CPU automatically if no GPU is available. Pass `device=cpu` to force
CPU. Prediction also works programmatically via
`saicinpainting.training.trainers.load_checkpoint`.

---

## 4. Training

```bash
export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)
python bin/train.py -cn lama-fourier \
    location=my_places \
    data.batch_size=8
```

Key points for a single AMD GPU:

- **Single GPU by default (important for dGPU + iGPU systems).** The trainer
  configs still say `gpus: -1` / `accelerator: ddp`; `bin/train.py` auto-translates
  these to the Lightning 2.x `accelerator/devices/strategy` API. Because many AMD
  desktops expose the integrated GPU as a second, compute-incapable device (multi-GPU
  DDP across a dGPU+iGPU pair crashes with an RCCL *"invalid device function"*),
  **`gpus: -1` now resolves to a single GPU** and prints a warning. You do **not**
  need `HIP_VISIBLE_DEVICES=0` or `trainer.kwargs.gpus=1` anymore. To train on
  several *real* compute GPUs, pass `trainer.kwargs.gpus=N` (N≥2) and select which
  devices with `HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES`.
- **`${env:VAR}` works out of the box.** The `env` OmegaConf resolver used by
  `resnet_pl.weights_path=${env:TORCH_HOME}` is registered automatically on import,
  so you no longer need to register it yourself. Just `export TORCH_HOME=...`.
- **Multi-worker data loading works.** `data.num_workers>0` no longer crashes with
  *"Trying to resize storage that is not resizable"* — dataset arrays are now made
  contiguous. Use e.g. `data.num_workers=8` for much faster training. `pin_memory`
  and `persistent_workers` are enabled automatically.
- **Faster matmuls.** `torch.set_float32_matmul_precision('high')` is set on import
  (override with the `LAMA_MATMUL_PRECISION` env var: `highest` for full fp32, or
  `off`). This silences the "Tensor Cores" warning and speeds up conv/matmul.
- The default `lama-fourier` config uses the **ResNet perceptual loss**
  (`resnet_pl`), which requires the ADE20K segmentation model under
  `$TORCH_HOME/ade20k/...`. Download it as described in the main README, or set
  `losses.resnet_pl.weight=0` to train without it.
- Start with `precision: 32` (the default). fp16/bf16 also work on ROCm but are
  more likely to surface numerical edge cases; enable them only once fp32 trains
  stably.

> **Custom training scripts** (e.g. a `train_bars.py`) should import the
> Lightning-compat helpers instead of copying them:
> ```python
> from saicinpainting.training.trainers.pl_compat import (
>     migrate_trainer_kwargs, migrate_checkpoint_kwargs)
> ```
> so they automatically pick up the single-GPU default and other fixes.

---

## 5. What changed vs. upstream

### Dependencies (`conda_env.yml`, `requirements.txt`)
- Python 3.6 → **3.10**; PyTorch 1.8/CUDA 10.2 → **2.8.0 + ROCm 6.4**;
  torchvision → **0.23.0**; PyTorch Lightning 1.2.9 → **2.5.6**.
- All scientific deps bumped to Python-3.10-compatible releases (numpy 1.26,
  scipy, pandas 2.0, scikit-image 0.21, scikit-learn 1.3, opencv 4.9, kornia
  ≥0.7.3, albumentations 1.3.1, hydra-core 1.3.2, webdataset ≥0.2).
- **Dropped `imgaug`** (unmaintained, incompatible with modern numpy) and
  **`tensorflow`** (only used by an optional TensorBoard-reporting script, not by
  training or inference).

### PyTorch Lightning 1.x → 2.x migration (the bulk of the code changes)
- **Manual optimization for the GAN.** Lightning 2.0 removed automatic
  optimization with multiple optimizers (`optimizer_idx`). The
  generator/discriminator now step manually in `training_step`
  (`self.automatic_optimization = False`, `self.manual_backward`, explicit
  `optimizer.step()`), with gradient clipping applied manually.
- **Removed hooks replaced:** `training_step_end` logic folded into
  `training_step`; `validation_epoch_end(outputs)` → `on_validation_epoch_end`
  with a `self.validation_step_outputs` buffer.
- **Trainer API:** `bin/train.py` translates the legacy Trainer kwargs
  (`gpus`, `accelerator=ddp`, `period`, `replace_sampler_ddp`, `terminate_on_nan`,
  `log_gpu_memory`, `resume_from_checkpoint`, …) to their 2.x equivalents and
  drops the ones that were removed.
- `trainer.num_processes` → `trainer.world_size`.

### Other API updates
- **`torch.load(..., weights_only=False)`** everywhere LaMa checkpoints / model
  weights are loaded (torch ≥2.6 changed the default to `weights_only=True`,
  which cannot unpickle LaMa's stored OmegaConf config).
- **albumentations:** `aug.py` rewritten to use the native `A.Affine` /
  `A.Perspective` transforms instead of the removed imgaug-backed
  `DualIAATransform`.
- **Hydra 1.3:** added `version_base=None` to every `@hydra.main`, removed the
  `# @package _group_` directives (now the default behaviour) and dropped the
  `.yaml` suffix from `config_name`.
- **webdataset:** `webdataset.Dataset` → `webdataset.WebDataset`.
- Replaced removed numpy alias `np.int` with `int`.

### Stability & speed fixes for real AMD runs
- **`env` resolver** re-registered on import (`saicinpainting/__init__.py`) so
  `${env:VAR}` configs resolve under OmegaConf ≥2.1 (which removed the built-in
  `env` resolver).
- **Multi-worker DataLoader crash fixed.** Dataset arrays are made C-contiguous
  (`np.ascontiguousarray`) so `num_workers>0` no longer triggers *"Trying to resize
  storage that is not resizable"* during collation. `pin_memory` and
  `persistent_workers` are enabled by default for speed.
- **Single-GPU default** on machines with an integrated GPU (avoids the dGPU+iGPU
  DDP crash); multi-GPU is opt-in via `trainer.kwargs.gpus=N`. The migration
  helpers now live in `saicinpainting/training/trainers/pl_compat.py` so custom
  scripts can share them.
- **`set_float32_matmul_precision('high')`** on import for faster matmuls
  (`LAMA_MATMUL_PRECISION` to override).
- **Hydra `_self_`** added to the training configs' `defaults:` lists to silence
  the *"Defaults list is missing _self_"* warning (behaviour unchanged).
- The optional *"Detectron v2 is not installed"* message is now a debug log, not a
  print.

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `torch.cuda.is_available()` is `False` | Check ROCm install, `amdgpu` driver, and that the user is in the `render`/`video` groups. Confirm the torch build is `+rocm6.4`. |
| `RuntimeError: ... no kernel image is available` | Card arch not in the wheel; try `export HSA_OVERRIDE_GFX_VERSION=11.0.0`. |
| `_pickle.UnpicklingError` / `weights_only` error loading a checkpoint | Make sure you are on this fork — all loaders pass `weights_only=False`. |
| Training crashes constructing `ResNetPL` | Download the ADE20K model into `$TORCH_HOME`, or set `losses.resnet_pl.weight=0`. |
| Hydra error about `config_name` / `_group_` | Make sure you pulled this fork's `configs/` and `bin/` changes. |
| `RCCL/NCCL ... invalid device function` at startup on a machine with an iGPU | Expected — DDP across dGPU+iGPU. This fork defaults to a single GPU; if you forced multi-GPU, set `HIP_VISIBLE_DEVICES=0`. |
| `Trying to resize storage that is not resizable` with `num_workers>0` | Fixed in this fork (contiguous arrays). If you see it in custom dataset code, wrap returned arrays in `np.ascontiguousarray`. |
| Lots of **MIOpen** warnings, slow first iterations | Normal: MIOpen compiles & caches conv kernels on first use. They disappear once the cache (`~/.cache/miopen`) is warm. Ensure that dir is writable; do not delete it between runs. |
| `Unsupported interpolation type env` | You're on an older copy — pull this fork; the `env` resolver is now auto-registered. |
