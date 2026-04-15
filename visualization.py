import os
import torch
from diffusers import DDPMPipeline
from torchvision.utils import make_grid, save_image
from torchvision import transforms

# =========================
# CONFIG
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_IMAGES = 10
NROW = 5  # grid layout (5x2)
SAVE_DIR = "grids"

MODEL_PATHS = {
    "original": "checkpoints/celeb/deletion/2026-04-02_02-01-17_ee2d0b59-af34-4884-ad7f-e88ba8f661ea/unlearned_model",
    "synthetic": "checkpoints/celeb/synthetic/2026-04-02_02-06-18_699e10a4-036f-481e-93db-82b6a770a7a1",
    "hybrid_50": "checkpoints/celeb/hybrid/2026-04-02_02-09-41_a8156722-4125-4c83-bac7-0924eab958ae",
    "hybrid_70_real": "checkpoints/celeb/hybrid70r/2026-04-02_02-14-44_519bccbf-758a-42f4-ab66-1625a7f55c5f",
    "hybrid_30_real": "checkpoints/celeb/hybrid30r/2026-04-02_02-18-15_eed052b5-0953-47b1-a08a-e9cc7e0bed55",
}

os.makedirs(SAVE_DIR, exist_ok=True)

to_tensor = transforms.ToTensor()

# =========================
# FUNCTION
# =========================

def generate_grid(model_name, model_path):
    print(f"\n🔹 Processing {model_name}")

    pipe = DDPMPipeline.from_pretrained(model_path)
    pipe = pipe.to(DEVICE)

    # Optional: deterministic results
    generator = torch.Generator(device=DEVICE).manual_seed(42)

    images = pipe(
        batch_size=NUM_IMAGES,
        num_inference_steps=50,
        generator=generator
    ).images

    # Convert to tensor
    image_tensors = torch.stack([to_tensor(img) for img in images])

    # Create grid
    grid = make_grid(image_tensors, nrow=NROW)

    # Save
    save_path = os.path.join(SAVE_DIR, f"{model_name}_grid.png")
    save_image(grid, save_path)

    print(f"✅ Saved grid: {save_path}")


# =========================
# MAIN
# =========================

def main():
    for name, path in MODEL_PATHS.items():
        generate_grid(name, path)

    print("\n🎉 All grids generated!")


if __name__ == "__main__":
    main()
