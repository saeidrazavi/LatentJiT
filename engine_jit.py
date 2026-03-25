import math
import sys
import os
import shutil

import torch
import numpy as np
import cv2

import util.misc as misc
import util.lr_sched as lr_sched
import torch_fidelity
import copy
import torchvision.utils as vutils


# === NEW: VAE imports for latent-space pipeline ===
from diffusers import AutoencoderKL
from denoiser import denormalize_latents

try:
    import wandb
except ImportError:
    wandb = None

VAE_HF_NAME = "REPA-E/e2e-vavae-hf"


def load_vae(device="cuda", dtype=torch.float32):
    """
    Load the pre-trained E2E VA-VAE and extract latent statistics from its config.
    Called once in main_jit.py at startup. The VAE is only used during evaluation
    (to decode generated latents into pixels for FID). The stats are used in both
    training (normalize raw latents) and evaluation (denormalize before decoding).
    """
    vae = AutoencoderKL.from_pretrained(VAE_HF_NAME).to(device=device, dtype=dtype)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # Per-channel mean/std are stored directly in the VAE config
    # (see https://huggingface.co/REPA-E/e2e-vavae-hf/blob/main/config.json)
    latent_mean = torch.tensor(vae.config.latents_mean, dtype=dtype, device=device).view(1, -1, 1, 1)
    latent_std = torch.tensor(vae.config.latents_std, dtype=dtype, device=device).view(1, -1, 1, 1)

    return vae, latent_mean, latent_std


def decode_latents_to_pixels(vae, latents):
    """
    Decode raw (denormalized) VAE latents to pixel images.
    Input:  latents (B, 32, 16, 16) in raw VAE scale
    Output: pixels  (B, 3, 256, 256) in [0, 1]
    """
    with torch.no_grad():
        decoded = vae.decode(latents).sample       # (B, 3, 256, 256) in [-1, 1]
    pixels = ((decoded + 1.0) / 2.0).clamp(0, 1)   # -> [0, 1]
    return pixels
# === END NEW ===

def generate_and_log_samples(model_without_ddp, vae, latent_mean, latent_std, epoch, args, _cached={}):

    """Generate a 4x4 grid of images on GPU and log to wandb."""

    model_without_ddp.eval()
    num_samples = 16
    device = next(model_without_ddp.parameters()).device

    # Cache fixed noise and labels on first call
    if 'noise' not in _cached:
        rng = torch.Generator(device='cpu').manual_seed(42)
        _cached['noise'] = torch.randn(num_samples, model_without_ddp.latent_channels,
                                       model_without_ddp.latent_size, model_without_ddp.latent_size,
                                       generator=rng).to(device)
        _cached['labels'] = (torch.arange(num_samples) % model_without_ddp.num_classes).to(device)

    # Override generate() to use fixed noise
    z = _cached['noise'].clone()
    labels = _cached['labels']

    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        z = model_without_ddp.ode_solver(z, labels)
    z = denormalize_latents(z.float(), latent_mean, latent_std)
    pixels = decode_latents_to_pixels(vae, z).detach().cpu()  # (9, 3, 256, 256) in [0,1]

    # Make 4x4 grid
    grid = vutils.make_grid(pixels, nrow=4, padding=2, normalize=False)

    # Log to wandb
    if wandb is not None and wandb.run is not None:
        wandb.log({
            "epoch": epoch,
            "samples": wandb.Image(grid.permute(1, 2, 0).numpy(), caption=f"Epoch {epoch}"),
        })

    model_without_ddp.train()
    print(f"  Logged 4x4 sample grid for epoch {epoch}")



def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (x, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # CHANGED: removed pixel normalization (div 255, *2 - 1)
        # x arrives as pre-normalized latent (32, 16, 16) from the dataloader
        x = x.to(device, non_blocking=True).to(torch.float32)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model(x, labels)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None:
            # Use epoch_1000x as the x-axis in TensorBoard to calibrate curves.
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)

        # NEW: wandb logging every iteration
        global_step = epoch * len(data_loader) + data_iter_step
        if misc.is_main_process() and getattr(args, 'use_wandb', False) and wandb is not None and wandb.run is not None:
            wandb.log({
                "train/loss": loss_value_reduce,
                "train/lr": lr,
                "train/epoch": epoch + data_iter_step / len(data_loader),
                "train/global_step": global_step,
            }, step=global_step)


# CHANGED: signature adds vae, latent_mean, latent_std
def evaluate(model_without_ddp, args, epoch, batch_size=64, log_writer=None,
             vae=None, latent_mean=None, latent_std=None):

    model_without_ddp.eval()
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    num_steps = args.num_images // (batch_size * world_size) + 1

    # Load VAE on demand if not passed from main
    if vae is None:
        vae, latent_mean, latent_std = load_vae(device="cuda")

    # Construct the folder name for saving generated images.
    # CHANGED: args.img_size -> args.latent_size
    save_folder = os.path.join(
        args.output_dir,
        "{}-steps{}-cfg{}-interval{}-{}-image{}-latent{}".format(
            model_without_ddp.method, model_without_ddp.steps, model_without_ddp.cfg_scale,
            model_without_ddp.cfg_interval[0], model_without_ddp.cfg_interval[1],
            args.num_images, args.latent_size
        )
    )
    print("Save to:", save_folder)
    if misc.get_rank() == 0 and not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # switch to ema params, hard-coded to be the first one
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = model_without_ddp.ema_params1[i]
    print("Switch to ema")
    model_without_ddp.load_state_dict(ema_state_dict)

    # ensure that the number of images per class is equal.
    class_num = args.class_num
    assert args.num_images % class_num == 0, "Number of images per class must be the same"
    class_label_gen_world = np.arange(0, class_num).repeat(args.num_images // class_num)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])

    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        start_idx = world_size * batch_size * i + local_rank * batch_size
        end_idx = start_idx + batch_size
        labels_gen = class_label_gen_world[start_idx:end_idx]
        labels_gen = torch.Tensor(labels_gen).long().cuda()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # CHANGED: renamed variable for clarity
            sampled_latents = model_without_ddp.generate(labels_gen)

        # CHANGED: denormalize + VAE decode instead of simple (x+1)/2 pixel denorm
        sampled_latents = denormalize_latents(
            sampled_latents.float(), latent_mean, latent_std
        )
        sampled_images = decode_latents_to_pixels(vae, sampled_latents)
        sampled_images = sampled_images.detach().cpu()

        torch.distributed.barrier()

        # distributed save images
        for b_id in range(sampled_images.size(0)):
            img_id = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
            if img_id >= args.num_images:
                break
            gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(img_id).zfill(5))), gen_img)

    torch.distributed.barrier()

    # back to no ema
    print("Switch back from ema")
    model_without_ddp.load_state_dict(model_state_dict)

    # compute FID and IS
    if log_writer is not None:
        # CHANGED: hardcoded to 256 (latent f16 always decodes to 256×256)
        fid_statistics_file = 'fid_stats/jit_in256_stats.npz'
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=None,
            fid_statistics_file=fid_statistics_file,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        # CHANGED: img_size -> latent_size in log key
        postfix = "_cfg{}_latent{}".format(model_without_ddp.cfg_scale, args.latent_size)
        log_writer.add_scalar('fid{}'.format(postfix), fid, epoch)
        log_writer.add_scalar('is{}'.format(postfix), inception_score, epoch)
        print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))
        shutil.rmtree(save_folder)

        # NEW: wandb logging for eval metrics
        if misc.is_main_process() and getattr(args, 'use_wandb', False) and wandb is not None and wandb.run is not None:
            wandb.log({
                "eval/fid": fid,
                "eval/inception_score": inception_score,
                "eval/epoch": epoch,
            })

    torch.distributed.barrier()