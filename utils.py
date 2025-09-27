import os
import json
import random
import numpy as np
from kornia.color import lab_to_rgb
import torch
from torch.amp import autocast
from tqdm import tqdm
# from matplotlib.lines import Line2D
# import matplotlib.pyplot as plt


def process_one_epoch(split, device, epoch, end_epoch, model, loader, optimizer, scaler, scheduler, weighted_charbonnier, lpips, ms_ssim, chromahue, awl):
    total_pixel_loss = total_percep_loss = total_ssim_loss = total_chroma_loss = total_hue_loss = total_loss = 0.0
    total_samples = 0

    # wrap loader in tqdm for live updates
    loop = tqdm(loader, desc=f"{split.capitalize()} Epoch {epoch}/{end_epoch}", unit='batch', leave=False)

    # Choose gradient context
    grad_ctx = torch.enable_grad if split == "train" else torch.no_grad

    for L_tensor, ab_tensor in loop:
        L_tensor, ab_tensor = L_tensor.to(device), ab_tensor.to(device)
        bs = L_tensor.size(0)
        
        if split == "train":
            optimizer.zero_grad(set_to_none=True)
        
        with grad_ctx():
            with autocast(device_type=device):
                outputs = model(L_tensor)
                outputs = torch.clamp(outputs, -1.0, 1.0)

                lab_pred = torch.cat([L_tensor, outputs], dim=1)
                lab_true = torch.cat([L_tensor, ab_tensor], dim=1)

                weighted_charbonnier_loss = weighted_charbonnier(outputs, ab_tensor)
                perceptual_loss = lpips(lab_pred, lab_true)
                ms_ssim_loss = ms_ssim(lab_pred, lab_true)
                chroma_fidelity_loss, hue_loss = chromahue(outputs, ab_tensor)

                loss = awl(weighted_charbonnier_loss, perceptual_loss, ms_ssim_loss, chroma_fidelity_loss, hue_loss)
                # loss = charbonnier_loss * lambda_charbonnier + lambda_lpips * perceptual_loss + lambda_ssim * ssim_loss + chroma_fidelity_loss * 0.05 + hue_loss * 0.02

        if split == "train":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        # Track
        total_pixel_loss += weighted_charbonnier_loss.item() * bs
        total_percep_loss += perceptual_loss.item() * bs
        total_ssim_loss += ms_ssim_loss.item() * bs
        total_chroma_loss += chroma_fidelity_loss.item() * bs
        total_hue_loss += hue_loss.item() * bs
        total_loss += loss.item() * bs
        total_samples += bs

        loop.set_postfix({
            "Pixel": f"{weighted_charbonnier_loss.item():.4f}",
            "LPIPS": f"{perceptual_loss.item():.4f}",
            "SSIM": f"{ms_ssim_loss.item():.4f}",
            "Chroma": f"{chroma_fidelity_loss.item():.4f}",
            "Hue": f"{hue_loss.item():.4f}",
            "Total": f"{loss.item():.4f}"
        })
    
    if split == "validation":
        scheduler.step(total_loss / total_samples)

    return {
        "epoch": epoch,
        "pixel_loss": total_pixel_loss / total_samples,
        "percep_loss": total_percep_loss / total_samples,
        "ssim_loss": total_ssim_loss / total_samples,
        "chroma_loss": total_chroma_loss / total_samples,
        "hue_loss": total_hue_loss / total_samples,
        "total_loss": total_loss / total_samples
    }


def log_losses(split, epoch, end, stats):
    print(f"\n{split.capitalize()} Epoch [{epoch}/{end}] Metrics:")
    print(f"  Pixel Loss      : {stats['pixel_loss']:.4f}")
    print(f"  LPIPS Loss      : {stats['percep_loss']:.4f}")
    print(f"  MS-SSIM Loss    : {stats['ssim_loss']:.4f}")
    print(f"  Chroma Loss     : {stats['chroma_loss']:.4f}")
    print(f"  Hue Loss        : {stats['hue_loss']:.4f}")
    print(f"  Total Loss      : {stats['total_loss']:.4f}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unfreeze_layer(model, layer_name):
    layer = getattr(model.encoder, layer_name)
    for param in layer.parameters():
        param.requires_grad = True


def get_prev_losses(train_path, val_path):
    train_loss, val_loss = [], []
    if os.path.exists(train_path) and os.path.exists(val_path):
        with open(train_path) as log:
            train_loss = json.load(log)
        with open(val_path) as log:
            val_loss = json.load(log)
    return train_loss, val_loss
    

def separate_no_decay_params(module):
    no_decay = []
    decay = []
    for n, p in module.named_parameters():
        if p.requires_grad:
            if 'bias' in n or 'bn' in n.lower() or 'batchnorm' in n.lower():
                no_decay.append(p)
            else:
                decay.append(p)
    return no_decay, decay


def _lab_to_rgb(lab_tensor):
    L = lab_tensor[:, 0:1] * 100.0
    ab = (lab_tensor[:, 1:] + 1) * 255.0 / 2 - 128.0
    rgb_tensor = lab_to_rgb(torch.cat([L, ab], dim=1))
    rgb_tensor = torch.clamp(rgb_tensor, 0.0, 1.0)
    return rgb_tensor
