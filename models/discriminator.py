# Python script that defines the Discriminator network for LFGAN
# Author: Sasidharan Mahalingam
# Date Created: Jan 16 2026

# Import the required packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Discriminator(nn.Module):
    """
    Class that defines the Discriminator for the LFGAN network
    """
    def __init__(self, img_size, ang_res=8):
        """
        Constructor for the Discriminator

        Arguments:
        ----------
        img_size - Image dimensions in (height, width, channels) format
        ang_res - Angular resolution dimension that have to be synthesized

        Returns:
        --------
        Nothing
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(1024, 2048, 4, 2, 1),
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(2048, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv1 = nn.Conv2d(512, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.lky1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.lky2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(128, 512, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.lky3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(512, 512, 3, padding=1)

        output_size = int(512 * (ang_res * img_size[0] / 32) * (ang_res * img_size[1] / 32))
        self.features_to_prob = nn.Sequential(
            # nn.Linear(output_size, output_size),
            nn.Linear(output_size, 1)
        )

    def forward(self, x):
        """
        Member function for the forward pass of the depth network

        Arguments:
        ----------
        input_data - Input tensor of shape [B, H, W, V, U, C]

        Returns:
        --------
        Returns the probability of discriminator
        """
        B, H, W, V, U, C = x.shape
        assert C == 3

        x = x.permute(0, 5, 1, 4, 2, 3).contiguous()
        x = x.view(B, 3, H * U, W * V)

        x = self.net(x)

        residual = x
        out = self.lky1(self.bn1(self.conv1(x)))
        out = self.lky2(self.bn2(self.conv2(out)))
        out = self.lky3(self.bn3(self.conv3(out)))

        x = out + residual

        x = self.conv4(x)

        x = x.view(B, -1)

        return self.features_to_prob(x)

