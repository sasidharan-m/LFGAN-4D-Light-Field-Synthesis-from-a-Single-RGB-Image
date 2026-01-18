# Python script that defines the EPI feature losses
# Author: Sasidharan Mahalingam
# Date Created: Jan 16 2026

# Import the required packages
import torch.nn.functional as F
from .VGGloss import VGGFeatureExtractor, normalizeVGG

def extractHorizontalEPI(lf):
    """
    Function that extracts the Horizontal EPIs

    Arguments:
    ----------
    lf - Input Light Field of shape [B, H, W, V, U, 3]

    Returns:
    --------
    Returns horizontal EPIs of shape [B*H*U, V, 3, W]
    """
    B, H, W, V, U, C = lf.shape
    # Permute: [B,H,U,W,V,C]
    epi_h = lf.permute(0,1,4,2,3,5).contiguous()
    # Merge batch, spatial vertical, angular vertical dims
    epi_h = epi_h.view(B*H*U, W, V, C)
    # For VGG: need [N,C,H,W] → treat V as H, W as W
    epi_h = epi_h.permute(0,3,2,1)  # [B*H*U, C, V, W]
    return epi_h

def extractVerticalEPI(lf):
    """
    Function that extracts the Veritcal EPIs

    Arguments:
    ----------
    lf - Input Light Field of shape [B, H, W, V, U, 3]

    Returns:
    --------
    Returns vertical EPIs of shape [B*W*V, U, 3, H]
    """
    B, H, W, V, U, C = lf.shape
    # Permute: [B,W,V,H,U,C]
    epi_v = lf.permute(0,2,3,1,4,5).contiguous()
    # Merge batch, spatial horizontal, angular horizontal dims
    epi_v = epi_v.view(B*W*V, H, U, C)
    # For VGG: need [N,C,H,W] → treat U as H, H as W
    epi_v = epi_v.permute(0,3,2,1)  # [B*W*V, C, U, H]
    return epi_v

def EPILoss(vgg, generated_lf, real_lf):
    """
    Function that calculates the EPI loss

    Arguments:
    ----------
    vgg - VGG model
    generated_lf - Generated Light Field
    real_lf - Ground truth Light Field

    Returns:
    --------
    Returns the EPI loss
    """
    generated_h = extractHorizontalEPI(generated_lf)
    real_h = extractHorizontalEPI(real_lf)

    generated_v = extractVerticalEPI(generated_lf)
    real_v = extractVerticalEPI(real_lf)

    generated_h = normalizeVGG(generated_h)
    real_h = normalizeVGG(real_h)
    generated_v = normalizeVGG(generated_v)
    real_v = normalizeVGG(real_v)

    generated_h_feats = vgg(generated_h.cpu())
    real_h_feats = vgg(real_h.cpu())
    generated_v_feats = vgg(generated_v.cpu())
    real_v_feats = vgg(real_v.cpu())

    loss_h = 0.0
    loss_v = 0.0
    for f_fake, f_real in zip(generated_h_feats, real_h_feats):
        loss_h += F.mse_loss(f_fake, f_real)

    for f_fake, f_real in zip(generated_v_feats, real_v_feats):
        loss_v += F.mse_loss(f_fake, f_real)

    return loss_h + loss_v
