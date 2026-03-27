import torch
import numpy as np
from diffusers import DDPMPipeline
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import pandas as pd

# Import your repo FID
from metrics.fid import FIDEvaluator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# CONFIG
# ----------------------------
BASE_MODEL_PATH = "google/ddpm-celebahq-256"
UNLEARNED_MODEL_PATH = "checkpoints/celeb/synthetic/2026-03-23_22-40-47_b06624a4-dac9-480c-bd83-2669dc422856"

DATASET_PATH = "data/datasets/celeba_hq_256"  # IMPORTANT: match your FID loader
NUM_SAMPLES = 1000
BATCH_SIZE = 8

# ----------------------------
# LOAD MODELS
# ----------------------------
def load_pipeline(path):
    pipe = DDPMPipeline.from_pretrained(path).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe

# ----------------------------
# IMAGE GENERATION
# ----------------------------
@torch.no_grad()
def generate_images(pipe, num_images):
    images = []
    for _ in tqdm(range(num_images // BATCH_SIZE)):
        batch = pipe(
            batch_size=BATCH_SIZE,
            num_inference_steps=50
        ).images
        batch = [transforms.ToTensor()(img) for img in batch]  # convert to tensor [0,1]
        batch = torch.stack(batch).to(device)
        images.append(batch)

    return torch.cat(images, dim=0)

# ----------------------------
# FID USING YOUR IMPLEMENTATION
# ----------------------------
def compute_fid(fake_images):
    print("Computing FID (repo version)...")

    fid_eval = FIDEvaluator(
        inception_batch_size=64,
        device=device
    )

    # Load real dataset (CelebA-HQ)
    fid_eval.load_celeb()

    # Add generated images
    fid_eval.add_fake_images(fake_images)

    fid_score = fid_eval.compute(verbose=True)
    return fid_score.item()

# ----------------------------
# SSCD (same as before)
# ----------------------------
from torchvision.models import inception_v3
from torch.nn.functional import adaptive_avg_pool2d

class SSCDModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = inception_v3(pretrained=True)
        self.model.fc = torch.nn.Identity()
        self.model.to(device).eval()

    def forward(self, x):
        x = adaptive_avg_pool2d(self.model(x), (1, 1))
        return x.view(x.size(0), -1)

def compute_sscd(fake_images, real_images):
    model = SSCDModel()

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
    ])

    def embed(imgs):
        feats = []
        for img in imgs:
            x = transform(img).unsqueeze(0).to(device)
            f = model(x)
            feats.append(f.cpu())
        return torch.cat(feats)

    fake_feat = embed(fake_images)
    real_feat = embed(real_images)

    sim = torch.mm(fake_feat, real_feat.T)
    sim = sim / (fake_feat.norm(dim=1, keepdim=True) * real_feat.norm(dim=1))

    return sim.mean().item()

# ----------------------------
# NLL (same)
# ----------------------------
@torch.no_grad()
def compute_nll(pipe, real_images):
    unet = pipe.unet
    scheduler = pipe.scheduler

    total_nll = 0

    for img in real_images:
        x0 = img.unsqueeze(0).to(device)

        t = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=device)
        noise = torch.randn_like(x0)

        xt = scheduler.add_noise(x0, noise, t)
        pred = unet(xt, t).sample

        loss = ((pred - noise) ** 2).mean()
        total_nll += loss.item()

    return total_nll / len(real_images)

# ----------------------------
# LOAD REAL IMAGES
# ----------------------------
def load_real_images(num_images):
    from data.src.celeb_dataset import CelebAHQ

    dataset = CelebAHQ(
        filter='all',
        data_path=DATASET_PATH,
        transform=transforms.ToTensor()
    )

    imgs = []
    for i in range(num_images):
        imgs.append(dataset[i])

    return torch.stack(imgs).to(device)

# ----------------------------
# MAIN
# ----------------------------
def evaluate():
    print("Loading models...")
    base_pipe = load_pipeline(BASE_MODEL_PATH)
    unlearned_pipe = load_pipeline(UNLEARNED_MODEL_PATH)

    print("Generating images...")
    base_images = generate_images(base_pipe, NUM_SAMPLES)
    unlearned_images = generate_images(unlearned_pipe, NUM_SAMPLES)

    print("Loading real images...")
    real_images = load_real_images(NUM_SAMPLES)

    # ----------------------------
    # METRICS
    # ----------------------------
    fid_base = compute_fid(base_images)
    fid_unlearned = compute_fid(unlearned_images)

    sscd_base = compute_sscd(base_images, real_images)
    sscd_unlearned = compute_sscd(unlearned_images, real_images)

    nll_base = compute_nll(base_pipe, real_images)
    nll_unlearned = compute_nll(unlearned_pipe, real_images)

    # ----------------------------
    # TABLE
    # ----------------------------
    df = pd.DataFrame({
        "Model": ["Base", "Unlearned"],
        "FID ↓": [fid_base, fid_unlearned],
        "SSCD ↓": [sscd_base, sscd_unlearned],
        "NLL ↓": [nll_base, nll_unlearned],
    })

    print("\n===== FINAL RESULTS =====")
    print(df.to_string(index=False))

    df.to_csv("evaluation_results.csv", index=False)


if __name__ == "__main__":
    evaluate()
