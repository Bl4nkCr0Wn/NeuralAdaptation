# Install necessary libraries
''' Some environment installations for jupiter (run manually on pc)
!pip install pytorch torchvision numpy tqdm
! pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3
!git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
%cd stylegan2-ada-pytorch
!wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
'''

# Import necessary libraries
import torch
import numpy as np
import PIL.Image
import dnnlib
import legacy
#import pickle
import os
import shutil
import time
from tqdm import tqdm
from google.colab import files
#from IPython.display import display  # Import display for showing images

import functools

# Load pre-trained StyleGAN2 model
def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(model_path, 'rb') as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
        G.forward = functools.partial(G.forward, force_fp32=True)# this is for using cpu instead of GPU
    return G, device

# Generate an image from a latent vector with specified resolution
def generate_image(G, z, device, resolution, psi):
    z = torch.from_numpy(z).to(device)
    img = G(z, None, truncation_psi=psi, noise_mode='const')
    img = (img + 1) * (255 / 2)
    img = img.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)
    img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
    img = img.resize((resolution, resolution), PIL.Image.LANCZOS)
    return img

# Modify specific parameters in the latent vector
def modify_latent(z, indices, value):
    for index in indices:
        z[0, index] += value
    return z

# Save the image
def save_image(img, path):
    img.save(path)

# Save the latent vector to a file
def save_latent_vector(z, path):
    np.savetxt(path, z)

# Main function
def main(model_path, base_save_path, resolution):
    start_time = time.time()

    G, device = load_model(model_path)
    latent_dim = G.z_dim

    results_dir = os.path.join('Results', base_save_path)
    os.makedirs(results_dir, exist_ok=True)

    n = int(input("Enter the number of parameters (even number >= 2): "))
    while n < 2 or n % 2 != 0:
        n = int(input("Please enter an even number greater than or equal to 2: "))

    theta_increment = float(input("Enter the theta increment (greater than 0 and up to 360): "))
    while theta_increment <= 0 or theta_increment > 360:
        theta_increment = float(input("Please enter a number greater than 0 and up to 360: "))

    theta_amount = int(input('Enter the number of images to generate for each theta increment: '))


    noise_variance_special = float(input("Enter the noise variance for special angles: "))
    noise_variance_other = float(input("Enter the noise variance for other angles: "))

    num_special_images = int(input("Enter the number of images for special angles: "))

    initial_value = float(input("Enter the initial value for all parameters: "))
    radius = float(input("Enter the radius factor: "))

    psi = 0.15

    param_indices = np.random.choice(latent_dim, n, replace=False)
    np.random.shuffle(param_indices)
    cos_indices = param_indices[:n // 2]
    sin_indices = param_indices[n // 2:]

    # General loop for theta generation
    theta_values = []
    current_theta = 135

    while True:
        theta_values.append(current_theta % 360)
        current_theta += theta_increment

        # Stop when we wrap around to or exceed the starting theta
        if current_theta % 360 == 135:
            break

    total_steps = len(theta_values)
    special_angles = [135, 315]
    special_steps = [angle for angle in special_angles if angle in theta_values]

    num_special_images_total = (num_special_images - theta_amount) * len(special_steps)
    total_images = (total_steps * theta_amount) + num_special_images_total

    # Single progress bar for the entire process
    progress_bar = tqdm(total=total_images, desc="Generating images")

    first_image_displayed = False

    for theta in theta_values:
        z = np.full((1, latent_dim), initial_value)

        values_cos = radius * np.cos(np.deg2rad(theta))
        values_sin = radius * np.sin(np.deg2rad(theta))

        z = modify_latent(z, cos_indices, values_cos)
        z = modify_latent(z, sin_indices, values_sin)

        # Apply noise variance to unselected parameters
        for idx in range(latent_dim):
            if idx not in param_indices:
                noise = np.random.normal(0, noise_variance_special if theta in special_angles else noise_variance_other)
                z[0, idx] += noise

        num_images = num_special_images if theta in special_angles else theta_amount
        for i in range(num_images):
            img = generate_image(G, z, device, resolution, psi)
            serial_number = str(progress_bar.n + 1).zfill(len(str(total_images)))
            theta_str = f"{theta:.1f}"
            img_path = os.path.join(results_dir, f"{serial_number}_image_{theta_str}_{i + 1}.png")
            save_image(img, img_path)

            latent_vector_path = os.path.join(results_dir, f"{serial_number}_image_{theta_str}_{i + 1}_latent_vector.txt")
            save_latent_vector(z, latent_vector_path)

            # # Display the first image generated
            # if not first_image_displayed:
            #     display(img)
            #     first_image_displayed = True

            progress_bar.update(1)  # Progress bar updates for each image

            if i < num_images - 1:
                z = np.full((1, latent_dim), initial_value)
                z = modify_latent(z, cos_indices, values_cos)
                z = modify_latent(z, sin_indices, values_sin)
                for idx in range(latent_dim):
                    if idx not in param_indices:
                        noise = np.random.normal(0, noise_variance_special if theta in special_angles else noise_variance_other)
                        z[0, idx] += noise

    progress_bar.close()

    parameter_details = [
        f"Selected parameters indices: {param_indices}",
        f"Cosine indices: {cos_indices}",
        f"Sine indices: {sin_indices}",
        f"Number of images generated: {total_images}",
        f"Noise variance for special angles: {noise_variance_special}",
        f"Noise variance for other angles: {noise_variance_other}",
        f"Psi: {psi}",
        f"Initial value for all parameters: {initial_value}",
        f"Radius factor: {radius}",
        f"User input: n={n}, theta_increment={theta_increment}, amount per degree={theta_amount} noise_variance_special={noise_variance_special}, noise_variance_other={noise_variance_other}"
    ]

    with open(os.path.join(results_dir, 'summary.txt'), 'w') as f:
        for detail in parameter_details:
            f.write(f"{detail}\n")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total running time: {elapsed_time:.2f} seconds")

    shutil.make_archive(results_dir, 'zip', results_dir)

    zipped_file_path = os.path.join('Results', f"{base_save_path}.zip")
    shutil.move(f"{results_dir}.zip", zipped_file_path)

    files.download(zipped_file_path)

# Parameters
model_path = 'ffhq.pkl'
base_save_path = 'generated_faces'
resolution = 512

if __name__ == "__main__":
    main(model_path, base_save_path, resolution)
