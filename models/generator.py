# Python script that defines the Generator network for LFGAN
# Author: Sasidharan Mahalingam
# Date Created: Jan 16 2026

# Import the required packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ConvBlock(nn.Module):
    """
    Class that defines the Conv Block used in the generator
    """
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
       """
        Constructor for the ConvBlock class
        
        Arguments:
        ----------
        in_c - number of input channels
        out_c - number of output channels
        k - kernel dimension
        s - stride
        p - padding

        Returns:
        --------
        Nothing
        """ 
       super().__init()
       self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p),
            nn.ReLU(),
            nn.BatchNorm2d(out_c)
       )

    def forward(self, x):
        """
        Member function for the forward pass of the depth network

        Arguments:
        ----------
        x - Input tensor

        Returns:
        --------
        Returns the output tensor
        """
        return self.block(x)

class ConvTransposeBlock(nn.Module):
    """
    Class that defines the ConvTranspose Block used for upsampling
    """
    def __init__(self, in_c, out_c, k=4, s=2, p=1):
        """
        Constructor for the ConvTransposeBlock class
        
        Arguments:
        ----------
        in_c - number of input channels
        out_c - number of output channels
        k - kernel dimension
        s - stride
        p - padding

        Returns:
        --------
        Nothing
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, k, s, p),
            nn.ReLU(),
            nn.BatchNorm2d(out_c)
        )
    
    def forward(self, x):
        """
        Member function for the forward pass of the depth network

        Arguments:
        ----------
        x - Input tensor

        Returns:
        --------
        Returns the output tensor
        """
        return self.block(x)
    
class ResBlock(nn.Module):
    """
    Class that defines the Residual Block used in the generator
    """
    def __init__(self, channels):
        """
        Constructor for the ResBlock class
        
        Arguments:
        ----------
        channels - number of channels used in the residual block

        Returns:
        --------
        Nothing
        """
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        """
        Member function for the forward pass of the depth network

        Arguments:
        ----------
        x - Input tensor

        Returns:
        --------
        Returns the output tensor
        """
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)



class Generator(nn.Module):
    """
    Class that defines the Generator for the LFGAN network
    """
    def __init__(self, ang_res=8):
        """
        Constructor for the Generator

        Arguments:
        ----------
        ang_res - Angular resolution dimension that have to be synthesized

        Returns:
        --------
        Nothing
        """
        super().__init__()
        self.U = ang_res
        self.V = ang_res

        # Residual Blocks
        self.res_blocks = nn.Sequential(*[ResBlock(64) for _ in range(16)])

        # Conv block 1
        self.conv_block1 = ConvBlock(64, 256)

        # Upsample block 1
        self.upsample_block1 = ConvTransposeBlock(256, 64)

        # Conv block 2
        self.conv_block2 = ConvBlock(64, 256)

        # Upsample block 2
        self.upsample_block2 = ConvTransposeBlock(256, 64)

        # Conv block 3
        self.conv_block3 = ConvBlock(64, 256)

        # Upsample block 3
        self.upsample_block3 = ConvTransposeBlock(256,64)

        # Final Conv block
        self.conv_block4 = ConvBlock(64, 3)

    def forward(self, x):
        """
        Member function for the forward pass of the depth network

        Arguments:
        ----------
        x - Input tensor of shape [B, H, W, 3]

        Returns:
        --------
        Returns the output tensor of shape [B, H, W, V, U, 3]
        """
        B, C, H, W = x.shape

        h = self.res_blocks(x)          # [B,H,W,64]
        h = self.conv_block1(h)         # [B,H,W,256]
        h = self.upsample_block1(h)     # [B,2*H,2*W,64]
        h = self.conv_block2(h)         # [B,2*H,2*W,256]
        h = self.upsample_block2(h)     # [B,4*H,4*W,64]
        h = self.conv_block3(h)         # [B,4*H,4*W,256]
        h = self.upsample_block3(h)     # [B,8*H,8*W,64]
        h = self.conv_block4(h)         # [B,8*H,8*W,64] -> [B,U*H,V*W,3]

        assert C == 3

        h = h.contiguous().view(B, H, self.U, W, self.V, 3)
        disp = h.permute(0, 1, 3, 4, 2, 5)

        return disp