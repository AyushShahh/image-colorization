import torch
import torch.nn as nn
import torch.nn.init as init
from torch.amp import autocast
import torchvision.models as models
from torchvision.models import ResNet34_Weights
import lpips
from utils import _lab_to_rgb
from pytorch_msssim import MS_SSIM


class ResNetEncoder(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()
        resnet = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            w = resnet.conv1.weight
            luminance = (0.2989*w[:, 0] + 0.5870*w[:, 1] + 0.1140*w[:, 2]).unsqueeze(1)
            self.conv1.weight.copy_(luminance)

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = (x - 0.449) / 0.226
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.maxpool(x)
        x2 = self.layer1(x1)
        del x1

        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return x, x2, x3, x4, x5


def icnr(tensor, scale=2, init_func=init.kaiming_normal_):
    ni, nf, h, w = tensor.shape
    ni2 = int(ni / (scale ** 2))
    k = init_func(torch.zeros([ni2, nf, h, w]))
    k = k.repeat_interleave(scale ** 2, 0)
    with torch.no_grad():
        tensor.copy_(k)


class PixelShuffleICNR(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (scale ** 2), kernel_size=3, padding=1)
        icnr(self.conv.weight, scale=scale)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.bn(x)
        x = self.gelu(x)
        # x = self.relu(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, d=(0.1, 0.1)):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            # nn.ReLU(inplace=True),
            nn.Dropout2d(p=d[0]),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            # nn.ReLU(inplace=True),
            nn.Dropout2d(p=d[1])
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)
    

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec4 = DecoderBlock(512, 256, 256, d=(0.18, 0.1))
        self.dec3 = DecoderBlock(256, 128, 128, d=(0.12, 0.06))
        self.dec2 = DecoderBlock(128, 64, 64, d=(0.08, 0.03))
        self.dec1 = DecoderBlock(64, 64, 64, d=(0.03, 0))
        self.pixel_shuffle = PixelShuffleICNR(64, 16, scale=2)
        self.final = nn.Conv2d(16, 2, kernel_size=3, padding=1)

    def forward(self, x5, x4, x3, x2, x1):
        d4 = self.dec4(x5, x4)
        d3 = self.dec3(d4, x3)
        del d4, x4, x3
        d2 = self.dec2(d3, x2)
        del d3, x2
        d1 = self.dec1(d2, x1)
        del d2, x1
        out = self.pixel_shuffle(d1)
        del d1
        out = self.final(out)
        return torch.tanh(out)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNetEncoder()
        self.decoder = Decoder()

    def forward(self, x):
        x, x2, x3, x4, x5 = self.encoder(x)
        return self.decoder(x5, x4, x3, x2, x)
    
    def train(self, mode=True):
        super().train(mode)
        for m in self.encoder.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                if not any(p.requires_grad for p in m.parameters()):
                    m.eval()

    def recalibrate_encoder_bn(self, loader, device="cuda", reset=True):
        for m in self.encoder.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                if reset:
                    m.reset_running_stats()
                m.momentum = None
                m.train()
        
        with torch.no_grad():
            for L, _ in loader:
                L = L.to(device, non_blocking=True)
                with autocast(device_type=device):
                    _ = self.encoder(L)
        
        for m in self.encoder.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()


class WeightedCharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        C_gt = torch.sqrt(target[:, 0]**2 + target[:, 1]**2 + self.eps**2)
        w = 1.0 + 1.23 * torch.clamp(C_gt - 0.05, min=0.0)  # 1–2.5x
        w = w.unsqueeze(1)
        diff = torch.sqrt((pred - target)**2 + self.eps**2)
        L_charb_weighted = torch.mean(w * diff)
        return L_charb_weighted  # / 7.5744
    

class LPIPSLoss(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.loss_fn = lpips.LPIPS(net=net, pretrained=True)
        self.loss_fn.eval()

        for param in self.loss_fn.parameters():
            param.requires_grad = False
    
    def forward(self, pred, target):
        pred = _lab_to_rgb(pred) * 2.0 - 1.0
        target = _lab_to_rgb(target) * 2.0 - 1.0
        return self.loss_fn(pred, target).mean()


class ChromaHueLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        # Compute chroma for both predicted and target images
        C_pred = torch.sqrt(pred[:, 0]**2 + pred[:, 1]**2 + self.eps)
        C_gt = torch.sqrt(target[:, 0]**2 + target[:, 1]**2 + self.eps)

        # Mask to avoid forcing color on near-gray regions
        mask = (C_gt > 0.05).float()

        # Chroma fidelity loss
        L_chroma_fid = torch.mean(mask * torch.sqrt((C_pred - C_gt)**2 + 1e-6))

        # Hue loss
        theta_p = torch.atan2(pred[:,1], pred[:,0])
        theta_g = torch.atan2(target[:,1], target[:,0])
        L_hue = torch.mean(mask * (1 - torch.cos(theta_p - theta_g)))

        return L_chroma_fid, L_hue # / 1.4132 and / 2.0
    

class MS_SSIMLoss(nn.Module):
    def __init__(self, data_range=1.0, size_average=True, channel=3):
        super().__init__()
        self.loss_fn = MS_SSIM(data_range=data_range, size_average=size_average, channel=channel)

    def forward(self, pred, target):
        pred = _lab_to_rgb(pred)
        target = _lab_to_rgb(target)
        return 1-self.loss_fn(pred, target)


class AutomaticWeightedLoss(nn.Module):
    """
    Implements automatic loss weighting.
    Kendall et al. 2017 (https://arxiv.org/abs/1705.07115)
    """
    def __init__(self, num_losses, clamp=(-4.0, 4.0)):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_losses))
        self.clamp_min, self.clamp_max = clamp

    def forward(self, *losses):
        loss_total = 0.0
        for i, loss in enumerate(losses):
            s = torch.clamp(self.log_vars[i], self.clamp_min, self.clamp_max)
            precision = torch.exp(-s)
            weighted_loss = (precision * loss + s) * 0.5
            loss_total += weighted_loss
        return loss_total
