import argparse
import torch
from diffusers import DDPMPipeline, UNet2DModel
import numpy as np
import json
from transformers import set_seed
from metrics.fid import FIDEvaluator

SEED = 42
INCEPTION_BATCH_SIZE = 256
IMGS_TO_GENERATE = 10000
GENERATION_BATCH_SIZE = 16

set_seed(SEED)

# Initialize the model ID and device
model_id = "google/ddpm-celebahq-256"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to parse CLI arguments
def parse_args():
    parser = argparse.ArgumentParser(description="FID Evaluation Script for DDPM Models")
    
    # Required argument for the output file path
    parser.add_argument('output_file', type=str, help="The JSON file to save FID scores")
    
    # Required argument for a list of checkpoint paths
    parser.add_argument('--checkpoints', nargs='+', required=True, type=str, help="List of checkpoint paths")

    return parser.parse_args()

def main():
    # Parse CLI arguments
    args = parse_args()
    
    checkpoints_paths = args.checkpoints
    fid_fpath = args.output_file

    # Load the base DDPM pipeline
    ddpm = DDPMPipeline.from_pretrained(model_id).to(device)

    # Initialize the FID evaluator
    fid_calculator = FIDEvaluator(inception_batch_size=INCEPTION_BATCH_SIZE, device=device)
    fid_calculator.load_celeb()

    # Dictionary to store FID scores
    fids = {}

    # Loop through each checkpoint, replace the UNet model, generate images, and compute FID
    for checkpoint_path in checkpoints_paths:
        print(f'Evaluating checkpoint path: {checkpoint_path}')

        # Load the checkpointed UNet model and replace it in the pipeline
        if checkpoint_path == 'original':
            ddpm = DDPMPipeline.from_pretrained(model_id).to(device)
            print('Loaded og model')
        else:
            ddpm.unet = UNet2DModel.from_pretrained(f'{checkpoint_path}/checkpoint-60/unet').to(device)

        # Initialize variables to track generated images
        num_batches = IMGS_TO_GENERATE // GENERATION_BATCH_SIZE
        remainder = IMGS_TO_GENERATE % GENERATION_BATCH_SIZE

        # Generate full batches and calculate FID incrementally
        for _ in range(num_batches):
            # Generate a batch of images
            images = ddpm(batch_size=GENERATION_BATCH_SIZE, num_inference_steps=50, output_type="numpy").images
            images = torch.from_numpy(images).permute(0, 3, 1, 2).to(device)

            # Add the generated images to the FID evaluator
            fid_calculator.add_fake_images(images)

        # Generate remaining images if there is a remainder
        if remainder > 0:
            images = ddpm(batch_size=remainder, num_inference_steps=50, output_type="numpy").images
            images = torch.from_numpy(images).permute(0, 3, 1, 2).to(device)

            # Add the generated images to the FID evaluator
            fid_calculator.add_fake_images(images)

        # Compute FID and store it in the dictionary
        fid = fid_calculator.compute(verbose=True).item()
        fids[checkpoint_path] = fid
        print(f'FID for checkpoint {checkpoint_path}: {fid}')

        # Save the FID scores to a JSON file
        with open(fid_fpath, 'w') as fid_file:
            json.dump(fids, fid_file, indent=4)

        print(f"FID scores saved to {fid_fpath}")

if __name__ == '__main__':
    main()
