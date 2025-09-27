import os
import json
import argparse
import torch
# from autoencoder import ED_CNN
from torch.amp import GradScaler
# from unet import UNet, VGGPerceptualLoss
from architecture import UNet, LPIPSLoss, WeightedCharbonnierLoss, ChromaHueLoss, MS_SSIMLoss, AutomaticWeightedLoss
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from dataset import load_coco_dataset
from utils import set_seed, unfreeze_layer, get_prev_losses, separate_no_decay_params, process_one_epoch, log_losses


def main():
    # Command line arguments
    parser = argparse.ArgumentParser(description="Train ResNet Autoencoder for Image Colorization")
    parser.add_argument("--start_epoch", type=int, default=1, help="Starting epoch for training")
    parser.add_argument("--end_epoch", type=int, default=80, help="Ending epoch for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=23, help="Set seed for reproducibility")
    parser.add_argument("--checkpoint", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--logs", type=str, default="logs", help="Directory for logs")
    parser.add_argument("--batchsize", type=int, default=32, help="Batch size for training")
    args = parser.parse_args()

    # Variables
    start_epoch = args.start_epoch
    end_epoch = args.end_epoch
    seed = args.seed
    batchsize = args.batchsize
    lr = args.lr

    checkpoint_path = f"{args.checkpoint}/last.pth"
    train_loss_path = f"{args.logs}/train_loss.json"
    val_loss_path = f"{args.logs}/val_loss.json"

    # Unfreezing schedules and layer learning rate factors
    unfreeze_schedule = {
        12: 'layer4',
        21: 'layer3',
        31: 'layer2'
    }
    layer_lrs = {
        'layer4': 0.05,
        'layer3': 0.02,
        'layer2': 0.01
    }

    # Setting seed and GPU backend
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    # Loading previous losses and dataset splits
    train_loss, val_loss = get_prev_losses(train_loss_path, val_loss_path)
    train_loader, val_loader = load_coco_dataset(batchsize)

    # Loading the model
    model = UNet().to(device)
    if start_epoch == 1:
        model.recalibrate_encoder_bn(train_loader, device=device)

    # Automatic weighted loss
    awl = AutomaticWeightedLoss(5).to(device)
    
    # Separating bias and batch norm parameters.
    # Weight decay is meant to shrink weights towards zero to prevent overfitting.
    # Applying it to Batch Normalization and other bias terms can be harmful.
    # https://arxiv.org/abs/2106.15739 (Lobacheva, Ekaterina, et al. 2021)
    # https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994
    decoder_no_weight_decay, decoder_params = separate_no_decay_params(model.decoder)

    # Initializing scaler, optimizer and scheduler
    scaler = GradScaler(device=device)
    optimizer = torch.optim.AdamW([
        {"params": decoder_params},
        {"params": decoder_no_weight_decay, "weight_decay": 0},
        {"params": awl.parameters(), "weight_decay": 0, "lr": lr * 0.1}
    ], lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-7
    )
    
    # Resuming from checkpoint if it exists
    if os.path.exists(checkpoint_path):
        print("Resuming from checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        awl.load_state_dict(checkpoint['awl'])

    # Unfreeze layers upto the current epoch in the schedule
    for e in unfreeze_schedule:
        if start_epoch <= e:
            break
        layer = unfreeze_schedule[e]
        unfreeze_layer(model, layer)
    
    # Load optimizer, scaler and scheduler states from the checkpoint
    if os.path.exists(checkpoint_path):
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    # Loss functions
    weighted_charbonnier = WeightedCharbonnierLoss().to(device)
    lpips = LPIPSLoss(net='vgg').to(device)
    ms_ssim = MS_SSIMLoss().to(device)
    chromahue = ChromaHueLoss().to(device)
    # l1_criterion = torch.nn.L1Loss()
    # criterion = VGGPerceptualLoss(layer_idx=16, device=device)
    # ssim = SSIMLoss(window_size=11, reduction='mean').to(device)

    # lambda_lpips = 0.2
    # lambda_ssim = 0.05
    # lambda_charbonnier = 0.75

    for epoch in range(start_epoch, end_epoch+1):
        model.train()

        if epoch in unfreeze_schedule:
            layer = unfreeze_schedule[epoch]
            unfreeze_layer(model, layer)
            new_params_no_weight_decay, new_params = separate_no_decay_params(getattr(model.encoder, layer))

            if epoch == 12:
                DEC_AFTER_WARMUP = min(3e-4, optimizer.param_groups[0]['lr'])
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[1]['lr'] = DEC_AFTER_WARMUP
                # lambda_charbonnier, lambda_lpips = 0.6, 0.35
            # elif epoch == 21:
                # lambda_charbonnier, lambda_lpips = 0.5, 0.45
            
            new_lr = layer_lrs[layer] * optimizer.param_groups[0]['lr']
            optimizer.add_param_group({"params": new_params, "lr": new_lr})
            optimizer.add_param_group({"params": new_params_no_weight_decay, "weight_decay": 0, "lr": new_lr})

        # Save losses
        epoch_stats = process_one_epoch("train", device, epoch, end_epoch, 
                                        model, train_loader, optimizer, scaler, scheduler,
                                        weighted_charbonnier, lpips, ms_ssim, chromahue, awl)

        train_loss.append(epoch_stats)
        os.makedirs(args.logs, exist_ok=True)
        with open(train_loss_path, "w") as f:
            json.dump(train_loss, f, indent=2)

        log_losses("train", epoch, end_epoch, epoch_stats)

        # validation loop
        model.eval()

        epoch_val_stats = process_one_epoch("validation", device, epoch, end_epoch,
                                            model, val_loader, optimizer, scaler, scheduler,
                                            weighted_charbonnier, lpips, ms_ssim, chromahue, awl)

        val_loss.append(epoch_val_stats)
        os.makedirs(args.logs, exist_ok=True)
        with open(val_loss_path, "w") as f:
            json.dump(val_loss, f, indent=2)

        log_losses("validation", epoch, end_epoch, epoch_val_stats)

        # Save model + states
        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'awl': awl.state_dict()
        }
        os.makedirs(args.checkpoint, exist_ok=True)
        torch.save(ckpt, f"{args.checkpoint}/epoch_{epoch:03}.pth")
        torch.save(ckpt, checkpoint_path)

        plot_losses()

    torch.save(model.state_dict(), "resnet_autoencoder.pth")


def plot_losses(save_path="plots/losses.png"):
    log_file="logs/train_loss.json"
    val_file = "logs/val_loss.json"

    os.makedirs("plots", exist_ok=True)
    with open(log_file) as f:
        log = json.load(f)

    with open(val_file) as f:
        val = json.load(f)

    epochs = [e['epoch'] for e in log]
    total = [e['total_loss'] for e in log]
    pixel = [e['pixel_loss'] for e in log]
    vgg = [e['percep_loss'] for e in log]
    ssim = [e['ssim_loss'] for e in log]

    val_total = [e['total_loss'] for e in val]
    val_pixel = [e['pixel_loss'] for e in val]
    val_vgg = [e['percep_loss'] for e in val]
    val_ssim = [e['ssim_loss'] for e in val]

    _, ax = plt.subplots(figsize=(10, 6))
    
    # Plot training losses (solid lines)
    ax.plot(epochs, total, color='blue', linestyle='-')
    ax.plot(epochs, pixel, color='orange', linestyle='-')
    ax.plot(epochs, vgg, color='green', linestyle='-')
    ax.plot(epochs, ssim, color='red', linestyle='-')
    
    # Plot validation losses (dashed lines)
    ax.plot(epochs, val_total, color='blue', linestyle='--')
    ax.plot(epochs, val_pixel, color='orange', linestyle='--')
    ax.plot(epochs, val_vgg, color='green', linestyle='--')
    ax.plot(epochs, val_ssim, color='red', linestyle='--')
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Validation Losses")
    
    # Get current y-axis limits
    y_min, y_max = ax.get_ylim()
    
    # Add extra padding at the top for legends (20% extra space)
    ax.set_ylim(y_min, y_max * 1.2)
    
    # Create legend handles
    loss_type_handles = [
        Line2D([0], [0], color='blue', label='Total'),
        Line2D([0], [0], color='orange', label='Charbonnier'),
        Line2D([0], [0], color='green', label='LPIPS/VGG'),
        Line2D([0], [0], color='red', label='SSIM')
    ]
    
    data_type_handles = [
        Line2D([0], [0], color='black', linestyle='-', label='Training'),
        Line2D([0], [0], color='black', linestyle='--', label='Validation')
    ]
    
    # Add legends with more optimal positioning
    first_legend = ax.legend(handles=loss_type_handles, loc='upper right', title="Loss Types", 
                           bbox_to_anchor=(1, 1), framealpha=0.9)
    ax.add_artist(first_legend)
    ax.legend(handles=data_type_handles, loc='upper left', title="Dataset", 
                            bbox_to_anchor=(0, 1), framealpha=0.9)
    
    # Adjust layout to make room for legends
    plt.tight_layout()
    
    ax.grid(True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # main()
    plot_losses()
