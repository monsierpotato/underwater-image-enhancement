# Physics-Guided Underwater Image Enhancement

U-Net trained with physics-guided input channels (UDCP transmission map + background light) on the EUVP benchmark.

Two model families are provided:

- **Deterministic U-Net** (`train.py`) — the standard 4-level U-Net (`unet_*` and the ResNet / MobileNet / Mamba backbones).
- **PUIE-UNet** (`PUIE-Unet.py`) — a **probabilistic** variant that grafts PUIE-Net's conditional-VAE mechanism (prior/posterior latent encoders + KL divergence, trained with an ELBO) onto the same deeper U-Net backbone. At inference it can run deterministically (MC) or average several prior samples (MP) for an ensembling boost.

---

## Environment Setup (Conda)

### 1. Create and activate the conda environment

```bash
conda create -n uwir python=3.10 -y
conda activate uwir
```

### 2. Install PyTorch (CUDA 12.1)

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

The editable install exposes the preferred commands `uwir-train`,
`uwir-evaluate`, `uwir-profile`, `uwir-puie-train`, and
`uwir-puie-evaluate`. The historical scripts remain available as compatibility
entry points.

> **Note**: PyTorch is already installed via conda, so pip will skip it.
> If pip tries to reinstall a CPU-only build, use
> `pip install -r requirements.txt --no-deps torch torchvision torchaudio` instead.

### 4. Verify the installation

```bash
python - <<'EOF'
import torch, torchvision, kornia, cv2, skimage, thop
print("torch     :", torch.__version__, "| CUDA:", torch.cuda.is_available())
print("torchvision:", torchvision.__version__)
print("kornia    :", kornia.__version__)
print("opencv    :", cv2.__version__)
print("scikit-img:", skimage.__version__)
print("thop      :", thop.__version__)
EOF
```

---

## Quick-start (one-liner after first-time setup)

```bash
conda activate uwir && python net_test.py
```

---

## Dataset Setup

Download datasets and place them under `./datasets/` with the following structure:

```
datasets/
  EUVP/
    Paired/
      underwater_imagenet/
        trainA/          ← degraded inputs
        trainB/          ← clean references
        testA/
        testB/
      underwater_dark/
        trainA/ trainB/ testA/ testB/
      underwater_scenes/
        trainA/ trainB/ testA/ testB/
  UIEB/
    raw-890/             ← degraded inputs
    reference-890/       ← reference images
  UFO120/
    train_val/
      lrd/               ← low-res / degraded
      hr/                ← high-quality reference
    test/
      lrd/
      hr/
  U45/                   ← 45 no-reference images (flat folder)
```

---

## Training

```bash
conda activate uwir

# Full 5-channel model (RGB + transmission map + background light)
python train.py --model unet_5ch --dataset euvp

# RGB-only baseline
python train.py --model unet_3ch --dataset euvp

# Combined EUVP + UIEB training
python train.py --model unet_5ch --dataset euvp+uieb

# Custom hyper-parameters
python train.py \
    --model unet_5ch \
    --dataset euvp \
    --batchSize 8 \
    --nEpochs 200 \
    --lr 1e-4 \
    --cos_restart True \
    --warmup_epochs 5 \
    --early_stop_patience 20
```

### Fast ASPP + Mamba architecture screen (Kaggle)

Three bottleneck-context ablations are available alongside the unchanged
U-Net baseline:

| Model prefix | Bottleneck context |
|---|---|
| `asppunet` | DeepLab-style ASPP |
| `mambabottleneck` | Two projected VSS/Mamba blocks at H/16 |
| `mambaaspp` | ASPP followed by the projected Mamba blocks |

The context branches are residual and retain the standard U-Net
encoder/decoder. This makes the comparison isolate context modelling rather
than changing the whole backbone.

In a Kaggle notebook, enable a GPU accelerator and run:

```bash
!git clone https://github.com/heniath/underwater-image-enhancement.git
%cd underwater-image-enhancement

# PyTorch is preinstalled on Kaggle. The fused scan is strongly recommended.
!pip install -e .
!pip install mamba-ssm --no-build-isolation

# Replace this with the actual EUVP directory shown under /kaggle/input.
!DATA_ROOT=/kaggle/input/euvp-dataset/EUVP \
  OUTPUT_ROOT=/kaggle/working/fast_screen \
  bash scripts/experiments/run_fast_context_screen.sh
```

The runner trains `unet_5ch`, `asppunet_5ch`,
`mambabottleneck_5ch`, and `mambaaspp_5ch` for 12 epochs on the same
`underwater_scenes` split and prints a validation ranking at the end. It does
not use the held-out test set for model selection.

Settings can be overridden without editing the script:

```bash
DATA_ROOT=/kaggle/input/.../EUVP \
BATCH_SIZE=4 EPOCHS=20 SEED=123 PRIOR_METHOD=gupdm \
bash scripts/experiments/run_fast_context_screen.sh
```

Promote a candidate when it gains roughly 0.20 dB validation PSNR without a
material SSIM loss. Confirm it with multiple seeds before a full-data run.

Key arguments (see `data/options.py` for the full list):

| Argument | Default | Description |
|---|---|---|
| `--model` | `unet_5ch` | Model variant (`unet_3ch`, `unet_5ch`, …) |
| `--dataset` | `euvp` | Training set (`euvp`, `uieb`, `ufo120`, `euvp+uieb`) |
| `--batchSize` | `16` | Mini-batch size |
| `--nEpochs` | `200` | Total epochs |
| `--lr` | `1e-4` | Initial learning rate |
| `--cos_restart` | `True` | Cosine annealing with restarts |
| `--warmup_epochs` | `3` | Linear LR warm-up epochs |
| `--early_stop_patience` | `20` | Early stopping patience (epochs) |
| `--checkpoint_dir` | `./checkpoints/` | Where to save `.pth` files |

Checkpoints are saved to `./checkpoints/` and the training history JSON to `./results/`.

---

## PUIE-UNet (Probabilistic variant)

`PUIE-Unet.py` combines two ideas:

- **From PUIE-Net** — a conditional VAE. A **prior** encoder sees only the (physics-augmented) input; a **posterior** encoder also sees the ground truth. Each branch produces two latent codes — a *mean* latent `u` and a *std* latent `s` — injected back into the decoder via FiLM modulation `InstanceNorm(feat) · |s| + u`. Training optimises an **ELBO** = reconstruction + `β · KL(posterior ‖ prior)`. At test time only the prior branch runs.
- **From this repo** — the deeper 4-level U-Net backbone, the physics front-end (UDCP `t(x)` + background light `B`), `CompositeLoss` (L1 + VGG + SSIM) as the reconstruction term, plus the dataset loaders, scheduler, early-stopping, checkpointing and metric suite (all reused, not re-written).

> The *backbone* part of `--model` is ignored here (the backbone is always PUIE-UNet); only the **channel variant** (`3ch` / `4ch_t` / `4ch_b` / `5ch`) is used to pick the physics front-end.

### Training

```bash
conda activate uwir

# 5-channel PUIE-UNet on UIEB, with KL annealing
python PUIE-Unet.py \
    --dataset uieb \
    --data_train_uieb ./datasets/UIEB \
    --model unet_5ch \
    --batchSize 8 \
    --nEpochs 200 \
    --kl_weight 1.0 \
    --kl_anneal_epochs 20 \
    --run_name puie_unet_uieb

# RGB-only PUIE-UNet on EUVP
python PUIE-Unet.py --dataset euvp --data_train_euvp ./datasets/EUVP --model unet_3ch
```

PUIE-specific arguments (in addition to all of `train.py`'s; see `data/options.py`):

| Argument | Default | Description |
|---|---|---|
| `--latent_dim` | `20` | Dimensionality of each (`u` / `s`) latent code |
| `--kl_weight` | `1.0` | Max weight `β` on the KL term of the ELBO |
| `--kl_anneal_epochs` | `20` | Linearly ramp `β` from 0 → `--kl_weight` over N epochs (`0` = off) |
| `--num_samples` | `1` | Prior samples averaged at validation-metric time (`1` = MC, `>1` = MP) |

The training log adds two columns, **Recon** and **KL**, so you can watch the ELBO terms separately. Checkpoints follow the same layout as `train.py` (`best_model.pth`, `last_model.pth`, `epoch_XXXX.pth`, `training_history.json`).

### Inference / Evaluation

`PUIE-Unet-test.py` loads a checkpoint, runs the prior branch, saves enhanced images, and (when ground truth is available) reports the full metric suite.

```bash
# Paired UIEB test set → save outputs + PSNR/SSIM/CIEDE2000/UCIQE/UIQM
python PUIE-Unet-test.py \
    --model unet_5ch \
    --resume ./checkpoints/puie_unet_uieb_XXXXXXXX_XXXXXX/best_model.pth \
    --data_val_uieb   ./datasets/UIEB/test/input \
    --data_valgt_uieb ./datasets/UIEB/test/reference \
    --num_samples 8 \
    --val_folder ./results/puie_unet_uieb

# No-reference folder (U45) → save outputs only
python PUIE-Unet-test.py \
    --model unet_3ch \
    --resume ./checkpoints/run/best_model.pth \
    --data_val_u45 ./datasets/U45 \
    --num_samples 8 \
    --val_folder ./results/puie_u45
```

- `--num_samples 1` → **MC** mode (deterministic, uses the prior means).
- `--num_samples N > 1` → **MP** mode (averages N prior samples — reduces variance, often a small PSNR/SSIM gain).

> **Important:** pass the same `--latent_dim` used during training (default `20`), otherwise the checkpoint's tensor shapes will not match.

---

## Evaluation

```bash
python eval.py
```

---

## Model Profiling

```bash
python net_test.py
```

Prints inference time, parameter count (M), and FLOPs (G) for a `256×256` input.

---

## Ablation Variants

| Variant | Input channels | Description |
|---|---|---|
| `unet_3ch` | 3 | RGB-only baseline |
| `unet_4ch_t` | 4 | RGB + transmission map t(x) |
| `unet_4ch_b` | 4 | RGB + background light B |
| **`unet_5ch`** | **5** | **RGB + t(x) + B ← proposed** |

---

## Project Structure

```
underwater-image-enhancement/
├── src/uwir/             — Installable library and command implementations
│   ├── data/             — Dataset classes, transforms, and factories
│   ├── models/           — Architectures and the central model registry
│   ├── physics/          — UDCP, GDCP, and GUPDM priors
│   ├── training/         — Schedulers and reusable training utilities
│   ├── cli/              — Training, evaluation, PUIE, and profiling commands
│   ├── config.py         — Typed configuration and CLI compatibility aliases
│   ├── losses.py         — Composite loss implementation
│   └── metrics.py        — Full- and no-reference underwater metrics
├── scripts/              — Experiments, visualization, and diagnostics
├── tests/                — Isolated pytest unit and smoke tests
├── data/, net/, loss/    — Compatibility imports for existing user code
├── train.py, eval.py     — Compatibility command wrappers
├── pyproject.toml        — Packaging, Ruff, pytest, and console commands
└── README.md
```

New code should import from `uwir`, for example:

```python
from uwir.data import EUVPDataset
from uwir.models import build_model, parse_model_variant
```

Legacy imports such as `from data.UWIRdataset import EUVPDataset` continue to
resolve to the same class.

## Development

```bash
pip install -e '.[dev]'
ruff format --check .
ruff check .
pytest
```

Generated datasets, checkpoints, logs, figures, reports, and temporary
artifacts are intentionally excluded from version control. Existing local
artifacts are not removed by the cleanup.

Runtime outputs are written beneath the current working directory by default:
`./checkpoints`, `./logs`, and `./results`. Override these with the relevant
CLI output-directory options when running from another location.

---

## Results

| Variant | Val PSNR | Test PSNR | Val SSIM | Test SSIM |
|---|---|---|---|---|
| `unet_3ch` (RGB only) | — | — | — | — |
| `unet_5ch` (proposed) | — | — | — | — |
| PUIE-UNet `unet_5ch` (MC) | — | — | — | — |
| PUIE-UNet `unet_5ch` (MP, 8 samples) | — | — | — | — |

*Fill in after training.*
