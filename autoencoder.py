import torch.nn as nn
import torch.nn.functional as F
import torch


class ED_CNN(nn.Module):
    def __init__(self):
        super(ED_CNN, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 2, kernel_size=3, padding=1, stride=2, output_padding=1)

    def forward(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = F.relu(self.conv3(x))
        x = self.maxpool3(x)

        # Decoder
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.tanh(self.deconv3(x))
        
        return x