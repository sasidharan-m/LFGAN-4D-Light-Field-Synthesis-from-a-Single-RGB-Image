# Python script that starts the training of the defined neural network pipeline that generates the lightfield given the central view
# Author: Sasidharan Mahalingam
# Date Created: Jan 16 2026

# Import the required packages
import os
import torch
from tqdm import tqdm

from data.dataloader import getDataloader
from models.generator import Generator
from models.discriminator import Discriminator
from losses.losses import gradientPenalty, generatorWGANLoss, generatorMSELoss
from losses.VGGloss import VGGPerceptualLoss, VGGFeatureExtractor
from losses.EPIloss import EPILoss
from losses.BRIloss import BRIloss

def generatorLoss(generated_lf, real_lf, D, vgg, vgg_loss_fn, epoch, device, lambda_adv=1e-3, lambda_mse=1.0, lambda_vgg=2e-6, lambda_epi=2e-6, lambda_bri=1.0):
    """
    Function that calculates the generator loss for LFGAN
    
    Arguments:
    ----------
    generated_lf - Generated Light Field data by the generator
    real_lf - Ground truth Light Field data
    D - Discriminator network
    vgg - VGG model
    vgg_loss_fn - VGG loss function
    epoch - Epoch number
    device - Device to run the computations on
    lambda_adv - weight for the Adversarial Loss
    lambda_mse - weight for the MSE loss
    lambda_vgg - weight for the VGG loss
    lambda_epi - weight for the EPI loss
    lambda_bri - weight for the BRI loss

    Returns:
    --------
    Returns the generator loss value
    """
    
    # WGAN-GP adversarial loss
    d_generated = D(generated_lf)
    adv_loss = torch.tensor([0.0], requires_grad=True, device=device)
    if(epoch > 200):
        adv_loss = generatorWGANLoss(d_generated)

    # MSE loss
    mse_loss = generatorMSELoss(real_lf, generated_lf)

    # VGG loss
    vgg_loss = torch.tensor([0.0], requires_grad=True, device=device)
    if(epoch > 200):
        vgg_loss = vgg_loss_fn(generated_lf.cpu(), real_lf.cpu())

    # EPI loss
    epi_loss = torch.tensor([0.0], requires_grad=True, device=device)
    if(epoch > 500):
        epi_loss = EPILoss(vgg, generated_lf, real_lf)

    # BRI loss
    bri_loss = torch.tensor([0.0], requires_grad=True, device=device)
    if(epoch > 700):
        bri_loss = BRIloss(generated_lf, real_lf)

    # Total generator loss
    total_loss = lambda_adv * adv_loss + lambda_mse * mse_loss + lambda_vgg * vgg_loss + lambda_epi * epi_loss + lambda_bri * bri_loss

    # Return components for logging
    return total_loss, {
        "adv_loss": adv_loss.item(),
        "mse_loss": mse_loss.item(),
        "vgg_loss": vgg_loss.item(),
        "epi_loss": epi_loss.item(),
        "bri_loss": bri_loss.item()
    }



def trainStepLFGAN(G, D, vgg, vgg_loss_fn, real_lf, input_img, g_optimizer, d_optimizer, epoch, device="cuda",
                   lambda_adv=1e-3, lambda_mse=1.0, lambda_vgg=2e-6, lambda_epi=2e-6, lambda_bri=1.0, lambda_gp=10.0, critic_iters=5):
    """
    Function that does one full LFGAN training step: discriminator + generator
    
    Arguments:
    ----------
    G - generator model
    D - discriminator model
    vgg = VGG model
    vgg_loss_fn - VGG Loss function
    real_lf - Ground truth Light Field of dimension [B,H,W,V,U,3]
    input_img - Center sub-aperture view [B,H,W,3] (center view input)
    g_optimizer - Optimizer for the Generator network
    d_optimizer - Optimizer for the Discriminator network
    epoch - Current epoch number
    device - Device to run the computations on
    lambda_adv - weight for the Adversarial Loss
    lambda_mse - weight for the MSE loss
    lambda_vgg - weight for the VGG loss
    lambda_epi - weight for the EPI loss
    lambda_bri - weight for the BRI loss
    lambda_gp - weight for the gradient penalty

    Returns:
    --------
    Nothing
    """

    # -----------------------------
    # 1. Train Discriminator (critic)
    # -----------------------------
    for _ in range(critic_iters):
        fake_lf = G(input_img).detach()  # detach for D update
        d_real = D(real_lf)
        d_fake = D(fake_lf)

        # WGAN-GP loss
        d_loss = d_fake.mean() - d_real.mean()  # WGAN discriminator loss
        gp = 0
        if(device == 'cpu'):
            gp = gradientPenalty(D, real_lf, fake_lf)
        else:
            gp = gradientPenalty(D, real_lf, fake_lf, use_cuda=True)
        
        d_total_loss = d_loss + lambda_gp * gp

        d_optimizer.zero_grad()
        d_total_loss.backward()
        d_optimizer.step()

    # -----------------------------
    # 2. Train Generator
    # -----------------------------
    fake_lf = G(input_img)
    g_total_loss, loss_dict = generatorLoss(fake_lf, real_lf, D, vgg, vgg_loss_fn, epoch, device,
                                            lambda_adv=lambda_adv, lambda_mse=lambda_mse, lambda_vgg=lambda_vgg, lambda_epi=lambda_epi, lambda_bri=lambda_bri)

    g_optimizer.zero_grad()
    g_total_loss.backward()
    g_optimizer.step()

    # -----------------------------
    # 3. Return losses for logging
    # -----------------------------
    loss_dict.update({
        "g_total_loss": g_total_loss.item(),
        "d_total_loss": d_total_loss.item(),
        "gp": gp.item()
    })
    return loss_dict

def trainLFGAN(training_data_path, weights_save_path, checkpoint_path="", grid=(13,13), crop_size=(256,256), batch_size=2, epochs=1000, learning_rate=1e-4, lambda_adv=1e-3, lambda_mse=1.0, lambda_vgg=2e-6, lambda_epi=2e-6, lambda_bri=1.0, lambda_gp=10.0, critic_iters=5, save_every=10):
    """
    Function that does the complete LFGAN training loop with model weight saving
    """
    loader = getDataloader(
        training_data_path, grid, spatial_crop=crop_size,
        batch_size=batch_size,
        resize=None,               
        num_workers=4
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator().to(device)
    discriminator = Discriminator((crop_size[0], crop_size[1], 3)).to(device)

    vgg = VGGFeatureExtractor()
    vgg_loss_fn = VGGPerceptualLoss(vgg)

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    checkpoint = None

    if checkpoint_path != "" and os.path.exists(checkpoint_path):
        print("Loading model state from checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        generator.load_state_dict[checkpoint["generator_state"]]
        g_optimizer.load_state_dict(checkpoint["generator_optim_state"])
        discriminator.load_state_dict[checkpoint["discriminator_state"]]
        d_optimizer.load_state_dict(checkpoint["discriminator_optim_state"])

    for epoch in range(epochs):
        epoch_loss = {
            "g_total_loss": 0.0,
            "d_total_loss": 0.0,
            "adv_loss": 0.0,
            "mse_loss": 0.0,
            "vgg_loss": 0.0,
            "epi_loss": 0.0,
            "bri_loss": 0.0,
            "gp": 0.0
        }

        loop = tqdm(enumerate(loader), total=len(loader), ncols=80, desc=f"Epoch [{epoch+1}/{epochs}]", leave=True, dynamic_ncols=False)

        for i, (aif_batch, lf_batch) in loop:
            # move to device
            aif_batch = aif_batch.to(device)          # [B,3,h,w]
            lf_batch  = lf_batch.to(device)           # [B,h,w,V,U,3]
            aif_batch = aif_batch.permute(0,2,3,1)       # [B,3,H,W] -> [B,H,W,3]

            # --- Training step ---
            loss_dict = trainStepLFGAN(
                generator, 
                discriminator,
                vgg,
                vgg_loss_fn,
                lf_batch, aif_batch,
                g_optimizer, d_optimizer,
                epoch,
                device=device,
                lambda_adv=lambda_adv,
                lambda_mse=lambda_mse,
                lambda_vgg=lambda_vgg,
                lambda_epi=lambda_epi,
                lambda_bri=lambda_bri,
                lambda_gp=lambda_gp,
                critic_iters=critic_iters
            )

            # Accumulate for epoch stats
            for k in epoch_loss:
                epoch_loss[k] += loss_dict[k]

        # Average losses for the epoch
        for k in epoch_loss:
            epoch_loss[k] /= batch_size

        print(f"Epoch [{epoch}/{epochs}] | "
              f"G_total={epoch_loss['g_total_loss']:.4f} | "
              f"D_total={epoch_loss['d_total_loss']:.4f} | "
              f"Adv={epoch_loss['adv_loss']:.4f} | "
              f"MSE={epoch_loss['mse_loss']:.4f} | "
              f"VGG={epoch_loss['vgg_loss']:.4f} | "
              f"EPI={epoch_loss['epi_loss']:.4f} | "
              f"BRI={epoch_loss['bri_loss']:.4f} | "
              f"GP={epoch_loss['gp']:.4f}")

        # -----------------------------
        # Save model checkpoints
        # -----------------------------
        if(((epoch + 1) % save_every == 0) and epoch > 1):
            ckpt = {
                "epoch": epoch + 1,
                "generator_state": generator.state_dict(),
                "generator_optim_state": g_optimizer.state_dict(),
                "discriminator_state": discriminator.state_dict(),
                "discriminator_optim_state": d_optimizer.state_dict(),
                "generator_loss": epoch_loss['g_total_loss'],
                "discriminator_loss": epoch_loss['d_total_loss'],
            }

            weights_name = f"checkpoint_epoch{epoch+1}.pth"
            torch.save(ckpt, os.path.join(weights_save_path, weights_name))

    g_path = os.path.join(weights_save_path, f"generator_weights.pt")
    d_path = os.path.join(weights_save_path, f"discriminator_weights.pt")
    torch.save(generator.state_dict(), g_path)
    torch.save(discriminator.state_dict(), d_path)


def main():
    """
    Driver function for training the model

    Parameters:
    -----------
    None

    Returns:
    --------
    Nothing
    """
    print("Starting training...")
    training_data_path = "/home/sasidharan/Projects/Plenoptic Camera/Datasets/Flower Dataset/Sub-Aperture Images/Train"
    weights_save_path = "/home/sasidharan/Projects/Plenoptic Camera/Code/LFGAN-4D-Light-Field-Synthesis-from-a-Single-RGB-Image/weights"
    trainLFGAN(training_data_path, weights_save_path, batch_size=1)
    print('Training Done.')

# Run the driver function
if __name__ == "__main__":
    main()