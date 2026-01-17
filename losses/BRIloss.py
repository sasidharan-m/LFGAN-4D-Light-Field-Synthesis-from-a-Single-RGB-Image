# Python script that defines the BRI feature losses
# Author: Sasidharan Mahalingam
# Date Created: Jan 16 2026

# Import the required packages
import torch

def calcHSPBrightness(lf):
    """
    Function to compute brightness using the HSP color model.

    Arguments:
    ----------
    lf - Input light field of dimension [B,H,W,V,U,3]

    Returns:
    --------
    Returns brightness map of dimension [B,H,W,V,U]
    """
    R = lf[..., 0]
    G = lf[..., 1]
    B = lf[..., 2]
    return torch.sqrt(0.299 * R**2 + 0.587 * G**2 + 0.114 * B**2)


def BRIloss(generated_lf, real_lf):
    """
    Function that calculates the BRI loss: L2 distance between brightness variance maps using HSP brightness
    Arguments:
    ----------
    generated_lf - Generated Light Field of dimension [B,H,W,V,U,3]
    real_lf - Ground truth Light Field of dimension [B,H,W,V,U,3]

    Returns:
    --------
    Returns the BRI loss value
    """
    # Compute brightness using HSP model
    generated_brightness = calcHSPBrightness(generated_lf)  # [B,H,W,V,U]
    real_brightness = calcHSPBrightness(real_lf)  # [B,H,W,V,U]

    # Compute variance along angular dimensions (V,U)
    generated_var = generated_brightness.var(dim=(-2,-1), unbiased=False)  # [B,H,W]
    real_var = real_brightness.var(dim=(-2,-1), unbiased=False)  # [B,H,W]

    # L2 distance between variance maps
    loss = torch.mean((generated_var - real_var)**2)
    return loss
