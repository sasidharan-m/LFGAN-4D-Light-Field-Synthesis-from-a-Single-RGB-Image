# Python script that defines the LFGAN losses
# Author: Sasidharan Mahalingam
# Date Created: Jan 16 2026

# Import the required packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import grad as torch_grad


def generatorWGANLoss(d_generated):
    """
    Function that defines the adversarial loss based on the WGAN-GP paper
    
    Arguments:
    ----------
    d_generated - Descriminator output for the generated light fields

    Returns:
    --------
    Returns the adversarial loss term
    """
    return -d_generated.mean()


def gradientPenalty(D, real_data, generated_data, use_cuda=False):
    """
    Function that defines the Gradient Penalty for the Generator network
    (Based on the WGAN-GP paper)
    
    Arguments:
    ----------
    D - Descriminator network
    real_data - Ground truth Light Field data
    generated_data: Generated Light Field data
    use_cuda: Flag that enables/disables cuda

    Returns:
    --------
    Returns the Gradient Penalty term
    """
    batch_size = real_data.size()[0]
    device = real_data.device

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1, 1, 1, 1, device=device)
    
    interpolated = alpha * real_data + (1 - alpha) * generated_data.detach()
    interpolated.requires_grad_(True)

    # Calculate probability of interpolated examples
    prob_interpolated = D(interpolated)
    prob_interpolated = prob_interpolated.view(batch_size, -1).mean(dim=1)

    grad_outputs = torch.ones_like(prob_interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                            grad_outputs=grad_outputs,
                            create_graph=True, only_inputs=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.reshape(batch_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return ((gradients_norm - 1) ** 2).mean()

def generatorMSELoss(real_data, generated_data):
    """
    Function that defines the MSE Loss for the Generator network

    Arguments:
    ----------
    real_data - Ground truth Light Field data
    generated_data - Generated Light Field data

    Returns:
    --------
    Returns the MSE loss term for the Generator
    """
    return F.mse_loss(generated_data, real_data)