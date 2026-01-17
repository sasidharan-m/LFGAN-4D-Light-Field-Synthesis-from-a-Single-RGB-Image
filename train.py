# Python script that starts the training of the defined neural network pipeline that generates the lightfield given the central view
# Author: Sasidharan Mahalingam
# Date Created: Jan 16 2026

# Import the required packages
import os
import torch

from losses.losses import gradientPenalty, generatorWGANLoss, generatorMSELoss
from losses.VGGloss import VGGPerceptualLoss
from losses.EPIloss import EPILoss
from losses.BRIloss import BRIloss

def generatorLoss(generated_lf, real_lf, D, device, lambda_adv=1e-3, lambda_mse=1.0, lambda_vgg=2e-6, lambda_epi=2e-6, lambda_bri=1.0):
    """
    Function that calculates the generator loss for LFGAN
    
    Arguments:
    ----------
    generated_lf - Generated Light Field data by the generator
    real_lf - Ground truth Light Field data
    D - Discriminator network
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
    adv_loss = generatorWGANLoss(d_generated)

    # MSE loss
    mse_loss = generatorMSELoss(real_lf, generated_lf)

    # VGG loss
    vgg_loss_fn = VGGPerceptualLoss().to(device)
    vgg_loss = vgg_loss_fn(generated_lf, real_lf)

    # EPI loss
    epi_loss = EPILoss(generated_lf, real_lf)

    # BRI loss
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



def trainStepLFGAN(G, D, real_lf, input_img, g_optimizer, d_optimizer, device="cuda",
                   lambda_adv=1e-3, lambda_mse=1.0, lambda_vgg=2e-6, lambda_epi=2e-6, lambda_bri=1.0, lambda_gp=10.0, critic_iters=5):
    """
    Function that does one full LFGAN training step: discriminator + generator
    
    Arguments:
    ----------
    real_lf - Ground truth Light Field of dimension [B,H,W,V,U,3]
    input_img - Center sub-aperture view [B,H,W,3] (center view input)
    g_optimizer - Optimizer for the Generator network
    d_optimizer - Optimizer for the Discriminator network
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
        gp = gradientPenalty(D, real_lf, fake_lf, device=device)
        d_total_loss = d_loss + lambda_gp * gp

        d_optimizer.zero_grad()
        d_total_loss.backward()
        d_optimizer.step()

    # -----------------------------
    # 2. Train Generator
    # -----------------------------
    fake_lf = G(input_img)
    g_total_loss, loss_dict = generatorLoss(fake_lf, real_lf, D, device, 
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

def trainLFGAN(G, D, dataloader, g_optimizer, d_optimizer, device="cuda", num_epochs=100, lambda_adv=1e-3, lambda_mse=1.0, lambda_vgg=2e-6, lambda_epi=2e-6, lambda_bri=1.0, lambda_gp=10.0, critic_iters=5, save_dir="weights", save_every=10):
    """
    Function that does the complete LFGAN training loop with model weight saving
    """
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, num_epochs+1):
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
        num_batches = 0

        for batch in dataloader:
            input_img = batch['center_view'].to(device)  # [B,3,H,W]
            real_lf = batch['lf'].to(device)             # [B,H,W,V,U,3]

            # --- Training step ---
            loss_dict = trainStepLFGAN(
                G, D,
                real_lf, input_img,
                g_optimizer, d_optimizer,
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
            num_batches += 1

        # Average losses for the epoch
        for k in epoch_loss:
            epoch_loss[k] /= num_batches

        print(f"Epoch [{epoch}/{num_epochs}] | "
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
        if epoch % save_every == 0 or epoch == num_epochs:
            g_path = os.path.join(save_dir, f"generator_epoch_{epoch}.pt")
            d_path = os.path.join(save_dir, f"discriminator_epoch_{epoch}.pt")

            torch.save(G.state_dict(), g_path)
            torch.save(D.state_dict(), d_path)

            print(f"Saved checkpoints at epoch {epoch}:")
            print(f"  Generator: {g_path}")
            print(f"  Discriminator: {d_path}")