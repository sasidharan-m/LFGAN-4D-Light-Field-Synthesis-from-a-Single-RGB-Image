# Python script that runs the trained LFGAN model
# Author: Sasidhran Mahalingam
# Date Created: Jan 21 2026

# Import the required packages
import os
import torch
from PIL import Image

from models.generator import Generator
import torchvision.transforms.functional as TF
from torchvision.utils import save_image


def test_LFGAN(generator_weights, input_image, output_dir):
    """
    Function that does the inference of the model to generate the light field data
    
    Parameters:
    -----------
    weights         - String that holds the path to the pre-trained weights
    input_image     - String that holds the path to the input image
    output_dir      - String that holds the path to save the generate light fields
    Returns:
    --------
    Nothing
    """
    os.makedirs(output_dir, exist_ok=True)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    generator = Generator().to(device)
    checkpoint = torch.load(generator_weights, map_location=device)
    generator.load_state_dict(checkpoint["generator_state"])
    generator.eval()

    # Prepare a batch of center views (aif_batch), e.g. from a DataLoader or single image
    img = Image.open(input_image).convert("RGB")
    img_t = TF.to_tensor(img).float().unsqueeze(0)
    img_t = img_t.permute(0,2,3,1)
    aif_batch = img_t.to(device)

    generated_lf = None
    # Infer
    with torch.no_grad():
        generated_lf = generator(aif_batch)

    # Save each sub-aperture view
    _, H, W, V, U, C = generated_lf.shape
    # Bring to [V*U, C, H, W] and [0,1] range
    views = generated_lf.squeeze(0)        # [H,W,V,U,3]
    views = views.permute(2, 3, 4, 0, 1)  # [V,U,3,H,W]
    views = views.reshape(V*U, C, H, W)

    # images are in [-1,1], map back to [0,1]
    views = (views + 1.0) * 0.5
    views = views.clamp(0,1)

    # save grid or individual files
    for idx in range(V*U):
        v = idx // U
        u = idx % U
        out_path = os.path.join(output_dir, 'sub-aperture_images')
        os.makedirs(out_path, exist_ok=True)
        out_path = os.path.join(output_dir, 'sub-aperture_images',f"view_{v}_{u}.png")
        save_image(views[idx], out_path)


def main():
    """
    Driver function for testing the model

    Parameters:
    -----------
    None

    Returns:
    --------
    Nothing
    """
    generator_weights = './weights/checkpoint_epoch10.pth'
    input_image = '/home/sasidharan/Projects/Plenoptic Camera/Datasets/Flower Dataset/Sub-Aperture Images/Test/IMG_0003/view_06_06.png'
    output_dir = './outputs'
    print('Starting Inference...')
    test_LFGAN(generator_weights, input_image, output_dir)
    print('Inference done. Synthesized Light Field data saved.')

# Run the driver function
if __name__ == "__main__":
    main()