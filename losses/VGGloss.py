# Python script that defines the VGG feature losses
# Author: Sasidharan Mahalingam
# Date Created: Jan 16 2026

# Import the required packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

class VGGFeatureExtractor(nn.Module):
    """
    Class that sets up the feature extraction using the VGG model
    """
    def __init__(self, layers=(3, 8, 15, 22)):
        """
        Constructor for the VGGFeatureExtractor class

        Argumments:
        -----------
        layers - layer dimensions corresponding to: relu1_2, relu2_2, relu3_3, relu4_3

        Returns:
        --------
        Nothing
        """
        super().__init__()
        vgg = vgg16(weights="IMAGENET1K_V1").features
        self.layers = layers

        self.blocks = nn.ModuleList()
        prev = 0
        for l in layers:
            self.blocks.append(nn.Sequential(*vgg[prev:l]))
            prev = l

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        """
        Member function for the forward pass of the depth network

        Arguments:
        ----------
        x - Input tensor [N, 3, H, W]

        Returns:
        --------
        Returns the output tensor
        """
        feats = []
        for block in self.blocks:
            x = block(x)
            feats.append(x)
        return feats
    

def normalizeVGG(x):
    """
    Function to normalize the VGG feature values

    Arguments:
    ----------
    x - Feature tensor of shape [N, 3, H, W]

    Returns:
    --------
    Returns the normalized output tensor of shape [N, 3, H, W]
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
    return (x - mean) / std

class VGGPerceptualLoss(nn.Module):
    """
    Class that does the VGGPerceptual Loss calculation
    """
    def __init__(self, vgg_feature_extractor, layer_weights=None):
        """
        Constructor for the class VGGPerceptualLoss

        Arguments:
        ---------
        vgg_feature_extractor - Object that holds the VGG model
        layer_weights - List containing the weights for different layers

        Returns:
        --------
        Nothing
        """
        super().__init__()
        self.vgg = vgg_feature_extractor
        self.layer_weights = layer_weights or [1.0, 1.0, 1.0, 1.0]

    def forward(self, generated_lf, real_lf):
        """
        Member function for the forward pass of the depth network

        Arguments:
        ----------
        generated_lf - Tensor of shape [B, H, W, V, U, 3] that has the generated Light Field data
        real_lf - Tensor of shape [B, H, W, V, U, 3] that has the real Light Field data

        Returns:
        --------
        Returns the output tensor of shape [B, H, W, V, U, 3]
        fake_lf, real_lf: [B, U*V, 3, H, W]
        """
        B, H, W, V, U, C = generated_lf.shape

        generated_lf_flat = generated_lf.permute(0, 3, 4, 1, 2, 5).reshape(B*V*U, C, H, W)
        real_lf_flat = real_lf.permute(0, 3, 4, 1, 2, 5).reshape(B*V*U, C, H, W)

        generated = normalizeVGG(generated_lf_flat)
        real = normalizeVGG(real_lf_flat)

        generated_feats = self.vgg(generated)
        real_feats = self.vgg(real)

        loss = 0.0
        for w, f_fake, f_real in zip(self.layer_weights, generated_feats, real_feats):
            loss += w * F.l1_loss(f_fake, f_real)

        return loss
