import torch
import numpy as np
from diffusers import DDPMPipeline
from torchvision import transforms
from tqdm import tqdm
from metrics.fid import FIDEvaluator
import torch.nn.functional as F
from data.src.celeb_dataset import CelebAHQ

# =========================
# SETTINGS
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"

NUM_SAMPLES = 1000
BATCH_SIZE = 8

# Disable gradients globally
torch.set_grad_enabled(False)

# =========================
# LOAD MODELS
# =========================

def load_model(path):
    pipe = DDPMPipeline.from_pretrained(path, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)
    return pipe

# =========================
# GENERATE IMAGES
# =========================

def generate_images(pipe):
    images = []
    total = 0
    pbar = tqdm(total=NUM_SAMPLES, desc="Generating Images")

    while total < NUM_SAMPLES:
        current_bs = min(BATCH_SIZE, NUM_SAMPLES - total)

        out = pipe(
            batch_size=current_bs,
            num_inference_steps=20
        ).images

        out = [transforms.ToTensor()(img) for img in out]
        images.extend(out)
        total += current_bs
        
        pbar.update(current_bs)

    pbar.close()
    return torch.stack(images)

# =========================
# FID
# =========================

def compute_fid(fid_eval, fake_images, reset=True):
    fid_eval.add_fake_images(fake_images.to(device))
    return fid_eval.compute(reset=reset).item()


# =========================
# SSCD
# =========================

class SSCD_Embedder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.fc = torch.nn.Identity()
        self.model.eval().to(device)

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device)[None, :, None, None]
        x = (x - mean) / std

        f = self.model(x)
        f = F.normalize(f, dim=1)
        return f

def compute_sscd(fake, real):
    model = SSCD_Embedder()

    def embed(x):
        feats = []
        for i in range(0, len(x), 32):
            batch = x[i:i+32].to(device)
            feats.append(model(batch).cpu())
        return torch.cat(feats)

    fake_f = embed(fake)
    real_f = embed(real)

    sim = fake_f @ real_f.T
    return sim.max(dim=1)[0].mean().item()

# =========================
# NLL
# =========================

def compute_nll(pipe, images):
    unet = pipe.unet
    scheduler = pipe.scheduler

    total_loss = 0
    count = 0

    for i in range(0, len(images), BATCH_SIZE):
        x = images[i:i+BATCH_SIZE].to(device).half()

        noise = torch.randn_like(x)
        t = torch.randint(
            0,
            scheduler.config.num_train_timesteps,
            (x.shape[0],),
            device=device
        )

        noisy = scheduler.add_noise(x, noise, t)
        pred = unet(noisy, t).sample

        loss = F.mse_loss(pred, noise, reduction='mean')
        total_loss += loss.item()
        count += 1

    return total_loss / count

# =========================
# LOAD REAL DATA
# =========================

def load_real_images(path):
    dataset = CelebAHQ(
        filter='all',
        data_path=path,
        transform=transforms.ToTensor()
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)

    images = []
    count = 0

    for x in loader:
        images.append(x)
        count += x.shape[0]

        if count >= NUM_SAMPLES:
            break

    return torch.cat(images)[:NUM_SAMPLES]

# =========================
# MAIN
# =========================

def evaluate():
    base_model_path = "google/ddpm-celebahq-256"
    unlearned_model_path = "checkpoints/celeb/hybrid/2026-04-02_02-09-41_a8156722-4125-4c83-bac7-0924eab958ae"

    base = load_model(base_model_path)
    unlearned = load_model(unlearned_model_path)

    print("Generating base model images...")
    base_imgs = generate_images(base)

    print("Generating unlearned model images...")
    unlearned_imgs = generate_images(unlearned)

    print("Loading real images...")
    real_imgs = load_real_images("data/datasets/celeba_hq_256")

    print("Computing FID...")

    fid_eval = FIDEvaluator(
        inception_batch_size=32,
        device=device
    )

    # Load real dataset ONCE
    fid_eval.load_celeb()

    # Base model
    fid_base = compute_fid(fid_eval, base_imgs, reset=False)

    # Unlearned model
    fid_unlearned = compute_fid(fid_eval, unlearned_imgs, reset=True)

    print("Computing SSCD...")
    sscd_base = compute_sscd(base_imgs, real_imgs)
    sscd_unlearned = compute_sscd(unlearned_imgs, real_imgs)

    print("Computing NLL...")
    nll_base = compute_nll(base, real_imgs)
    nll_unlearned = compute_nll(unlearned, real_imgs)

    print("\n===== FINAL RESULTS =====")
    print(f"{'Metric':<15} {'Base':<15} {'Unlearned':<15}")
    print("-" * 45)
    print(f"{'FID':<15} {fid_base:<15.4f} {fid_unlearned:<15.4f}")
    print(f"{'SSCD':<15} {sscd_base:<15.4f} {sscd_unlearned:<15.4f}")
    print(f"{'NLL':<15} {nll_base:<15.4f} {nll_unlearned:<15.4f}")

if __name__ == "__main__":
    evaluate()
