# LatentJiT: Latent-Space Diffusion with Just image Transformer

This repository adapts [JiT (Just image Transformer)](https://arxiv.org/abs/2511.13720) from pixel-space diffusion to **latent-space diffusion**, using pre-computed latent features from [REPA-E](https://github.com/thu-ml/REPA)'s VA-VAE.

Instead of training on raw 256×256 ImageNet pixels, LatentJiT trains on compact **16×16×32** latent representations pre-extracted by VA-VAE, and decodes generated latents back to pixels using the same pretrained VA-VAE decoder at evaluation time.

## Key Modifications from Original JiT

- **Latent-space input/output**: The model operates on 16×16×32 latent tensors instead of 256×256×3 pixel images.
- **Pre-computed latents**: Training uses the [G-REPA/imagenet-latents-davae-vavae-align1.5-400k](https://huggingface.co/datasets/G-REPA/imagenet-latents-davae-vavae-align1.5-400k) dataset (~93GB, 1.28M samples) — no VAE encoding needed.
- **VA-VAE decoder**: The pretrained [REPA-E/e2e-vavae-hf](https://huggingface.co/REPA-E/e2e-vavae-hf) VAE is used only at evaluation time for latent → pixel decoding.
- **Per-channel normalization**: Latents are normalized using mean/std statistics shipped in the VA-VAE config (`latents_mean`, `latents_std`).
- **Architecture**: `LatentEmbed` (Linear 32→D) replaces `PatchEmbed`, `FinalLayer` outputs 32 channels, and unpatchify is a simple reshape. The DiT transformer core is unchanged.

## Model Variants

| Model | Parameters | Embed Dim | Depth | Heads |
|-------|-----------|-----------|-------|-------|
| LatentJiT-S | 32M | 384 | 12 | 6 |
| LatentJiT-B | 130M | 768 | 12 | 12 |
| LatentJiT-L | 458M | 1024 | 24 | 16 |
| LatentJiT-XL | 675M | 1152 | 28 | 16 |

## Installation

```bash
# Clone the repository
git clone https://github.com/G-REPA/LatentJiT.git
cd latentjit

# Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

#initialize
uv init
 
# Install dependencies
uv add -r requirements.txt
uv sync
```

# Dataset

LatentJiT uses pre-computed ImageNet latent features extracted by REPA-E's VA-VAE, hosted on HuggingFace. Each sample contains a `data` field of shape `(64, 16, 16)` where the first 32 channels are the posterior mean and the last 32 are the posterior std. By default, LatentJiT uses the posterior mean only (`--use_mean_only`).

### Preparing the dataset

Run the provided `prepare_dataset.py` script to download and organize all required files:

```bash
uv run prepare_dataset.py
```

This script will:
1. Download Arrow shards, JSON metadata, and validation data from the [G-REPA/imagenet-latents-davae-vavae-align1.5-400k](https://huggingface.co/datasets/G-REPA/imagenet-latents-davae-vavae-align1.5-400k) HuggingFace dataset.
2. Organize root-level `.arrow` and `.json` files into the `train/` subfolder.
3. Download the ImageNet labels file from [G-REPA/imagenet_labels](https://huggingface.co/datasets/G-REPA/imagenet_labels).

### Expected Dataset Structure

```
ImageNet-Latents/
├── train/
│   ├── data-00000-of-00180.arrow
│   ├── data-00001-of-00180.arrow
│   ├── ... (all 180 shards)
│   ├── dataset_info.json          
│   └── state.json                
├── val/
│   ├── data-00000-of-00004.arrow
│   ├── ... (all 7 shards)
│   ├── dataset_info.json
│   └── state.json 
└── imagenet_train_labels.txt     
```


The pretrained VA-VAE decoder is automatically downloaded from [REPA-E/e2e-vavae-hf](https://huggingface.co/REPA-E/e2e-vavae-hf) during evaluation.

## Training

### Single GPU

```bash
uv run main_jit.py \
    --model LatentJiT-S \
    --batch_size 128 --blr 5e-5 \
    --epochs 200 --warmup_epochs 5 \
    --use_mean_only \
    --output_dir ./output_latentjit_s \
    --resume ./output_latentjit_s
```

### Multi-GPU (4× GPUs)

**LatentJiT-S (32M):**
```bash
uv run main_jit.py \
    --model LatentJiT-S \
    --batch_size 128 --blr 5e-5 \
    --epochs 200 --warmup_epochs 5 \
    --use_mean_only \
    --output_dir ./output_latentjit_s \
    --resume ./output_latentjit_s \
    --use_wandb
```

**LatentJiT-B (130M):**
```bash
uv run main_jit.py \
    --model LatentJiT-B \
    --batch_size 100 --blr 5e-5 \
    --epochs 200 --warmup_epochs 5 \
    --use_mean_only \
    --output_dir ./output_latentjit_b \
    --resume ./output_latentjit_b \
    --use_wandb
```

### With online evaluation

Computes FID every `--eval_freq` epochs during training:

```bash
uv run main_jit.py \
    --model LatentJiT-B \
    --batch_size 100 --blr 5e-5 \
    --epochs 200 --warmup_epochs 5 \
    --use_mean_only \
    --online_eval --eval_freq 40 \
    --num_images 50000 --gen_bsz 256 \
    --output_dir ./output_latentjit_b \
    --resume ./output_latentjit_b \
    --use_wandb
```

### Key training arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `LatentJiT-B` | Model variant (S/B/L/XL) |
| `--batch_size` | `128` | Batch size per GPU |
| `--blr` | `5e-5` | Base learning rate |
| `--epochs` | `200` | Total training epochs |
| `--P_mean` | `-0.8` | Logit-normal mean for timestep sampling |
| `--P_std` | `0.8` | Logit-normal std for timestep sampling |
| `--label_drop_prob` | `0.1` | Label dropout probability for CFG training |
| `--use_mean_only` | `False` | Use posterior mean only (no sampling) |
| `--hf_dataset` | HF hub ID | Path to local arrow files or HF dataset ID |
| `--use_wandb` | `False` | Enable Weights & Biases logging |

## Evaluation

### Full evaluation (50K images)

```bash
uv run main_jit.py \
    --model LatentJiT-B \
    --evaluate_gen \
    --resume ./output_latentjit_b \
    --num_images 50000 --gen_bsz 256 \
    --sampling_method heun --num_sampling_steps 50 \
    --cfg 1.0
```

### Quick sanity check (1K images)

```bash
uv run main_jit.py \
    --model LatentJiT-B \
    --evaluate_gen \
    --resume ./output_latentjit_b \
    --num_images 1000 --gen_bsz 64 \
    --sampling_method euler --num_sampling_steps 10
```

### Key evaluation arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--cfg` | `1.0` | Classifier-free guidance scale |
| `--sampling_method` | `euler` | Sampling method (`euler` or `heun`) |
| `--num_sampling_steps` | `50` | Number of denoising steps |
| `--num_images` | `50000` | Number of images to generate for FID |

## Logging

**TensorBoard** is enabled by default. Logs are saved to `{output_dir}/log/`:
```bash
tensorboard --logdir ./output_latentjit_b/log
```

**Weights & Biases** is enabled with `--use_wandb`:
```bash
wandb login  # one-time setup
# Then add --use_wandb to your training command
```

## FID Reference Statistics

FID is computed against ImageNet reference statistics. Use `prepare_ref.py` to generate reference stats from an ImageNet validation folder:

```bash
uv run python prepare_ref.py --img_dir /path/to/imagenet/val --out fid_stats/jit_in256_stats.npz
```

## Acknowledgements

This project builds on [JiT](https://github.com/LTH14/JiT) by Tianhong Li and Kaiming He, and uses the VA-VAE from [REPA-E](https://github.com/thu-ml/REPA) for latent encoding/decoding.

```bibtex
@article{li2025jit,
  title={Back to Basics: Let Denoising Generative Models Denoise},
  author={Li, Tianhong and He, Kaiming},
  journal={arXiv preprint arXiv:2511.13720},
  year={2025}
}
```

