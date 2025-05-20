import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from diffusers import DDPMPipeline, UNet2DModel
import json

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to load a pretrained model
def load_model(model_id="google/ddpm-celebahq-256", unet_path=None):
    ddpm = DDPMPipeline.from_pretrained(model_id).to(device)
    if unet_path:
        ddpm.unet = UNet2DModel.from_pretrained(unet_path).to(device)
    return ddpm

# Function to load an image and apply transforms
def load_image(image_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    image = Image.open(image_path)
    image = transform(image)
    return image.to(device)

# Function to inject noise at a specific timestep
def inject_noise(image, timestep, scheduler):
    noise = torch.randn_like(image, device=device)
    timestep_tensor = torch.tensor([timestep], device=device)  # Convert timestep to a tensor
    noisy_image = scheduler.add_noise(image, noise, timestep_tensor)
    return noisy_image, noise

# Function to denoise an image
def denoise(noisy_image, timestep, scheduler, model, batch_size=16):
    model_input = noisy_image.repeat(batch_size, 1, 1, 1)
    with torch.no_grad():
        for t in tqdm(reversed(range(timestep + 1))):
            model_output = model(model_input, t)["sample"]
            model_input = scheduler.step(model_output, t, model_input)["prev_sample"]
    return model_input

# Function to compute SSCD embeddings and similarity
def compute_sscd(denoised_image, train_img, sscd_model_path):
    model = torch.jit.load(sscd_model_path).to(device)
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
    )
    sscd_transform = transforms.Compose([
        transforms.Normalize(mean=[-1], std=[2]),
        normalize
    ])

    with torch.no_grad():
        denoised_transformed = sscd_transform(denoised_image)
        denoised_embedding = model(denoised_transformed)

        train_transformed = sscd_transform(train_img.unsqueeze(0))
        train_embedding = model(train_transformed)

    sscds = (train_embedding @ denoised_embedding.T)
    return sscds

# Function to display images
def show_image(image_tensors, rows=None, cols=None):
    def flatten(arr):
        flattened_arr = []
        for elem in arr:
            if isinstance(elem, list):
                flattened_arr.extend(elem)
            else:
                flattened_arr.append(elem)
        return flattened_arr
    
    if not isinstance(image_tensors, list):
        image_tensors = [image_tensors]
    
    image_tensors = [list(image_tensor) if image_tensor.dim() == 4 else image_tensor for image_tensor in image_tensors]
    image_tensors = flatten(image_tensors)
    num_images = len(image_tensors)
    
    if rows is None and cols is None:
        rows = int(math.sqrt(num_images))
        cols = int(math.ceil(num_images / rows))
    elif rows is None:
        rows = int(math.ceil(num_images / cols))
    elif cols is None:
        cols = int(math.ceil(num_images / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    
    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.flat
    
    for i, ax in enumerate(axes):
        if i < num_images:
            img_permuted = (image_tensors[i].squeeze().permute(1, 2, 0) + 1) / 2
            img_np = img_permuted.cpu().numpy()
            ax.imshow(img_np)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Main function to generate samples, denoise, and compute SSCD
def process_image(img_path, unet_path=None, sscd_model_path="checkpoints/classifiers/sscd_disc_mixup.torchscript.pt", timestep=250):
    # Load model and image
    model = load_model(unet_path=unet_path)
    scheduler = model.scheduler
    train_img = load_image(img_path)
    
    # Inject noise and show noisy image
    noisy_image, noise = inject_noise(train_img, timestep, scheduler)
    print(f"Noisy Image at Timestep {timestep}:")
    show_image(noisy_image)
    
    # Denoise image
    denoised_image = denoise(noisy_image, timestep, scheduler, model.unet, batch_size=16)
    
    # Show noisy, original, and denoised images
    show_image([noisy_image, train_img, denoised_image])

    # Compute SSCD similarity score
    sscds = compute_sscd(denoised_image, train_img, sscd_model_path)
    print(f'SSCDs: {sscds}')
    print(f'Mean SSCD Score: {sscds.mean()}')
    return sscds

if __name__ == "__main__":
    # Usage example
    # Call process_image with the path to your image and the UNet model path
    img_names = ['10000.jpg', '10001.jpg', '10002.jpg', '10003.jpg', '10004.jpg', '10005.jpg']

    for img_name in img_names:
        print(f'\nProcessing {img_name}')
        process_image(f'data/examples/celeba_hq_256/{img_name}', unet_path=None, sscd_model_path="checkpoints/classifiers/sscd_disc_mixup.torchscript.pt")


