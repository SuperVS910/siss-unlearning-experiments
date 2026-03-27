#SISS code reproduction and different experiments conducted using a synthetic, and hybrid dataset as the retain dataset

## Setup

1. Clone the repo.
    ```sh
    git clone https://github.com/SuperVS910/siss-unlearning-experiments.git
    cd siss-unlearning-experiments
    ```


2. Environment Setup
    ```sh
    conda activate siss
    ```

3. Download pretrained checkpoints and datasets.
    ```sh
    curl -L -o checkpoints.zip https://www.kaggle.com/api/v1/datasets/download/kenhas/data-unlearning-in-diffusion-models-checkpoints
    unzip checkpoints.zip

    curl -L -o datasets.zip https://www.kaggle.com/api/v1/datasets/download/kenhas/data-unlearning-in-diffusion-models-datasets
    unzip datasets.zip
    ```

4. After creating and activating the environment, you can run the wandb-compatible experiments with Hydra as follows:
    ```sh
    python main.py --config-name=[delete_celeb, delete_celeb_synthetic, delete_celeb_hybrid]
    ```

### Unlearning Methods (``losses/ddpm_deletion_loss.py``)

1. **SISS (importance_sampling_with_mixture)**
   - Uses a defensive mixture for importance sampling (one forward pass per batch).
   - **Key hyperparams**:
     - `lambd ∈ [0, 1]`: Balances sampling between kept data and unlearn set.
     - `scaling_norm`: Clips NegGrad term's gradient to this norm. Recommended to be tuned to 10% of naive deletion term's gradient norm.

### Monitored Metrics

- **FID**: Measures image quality for CelebA.
- **SSCD**: Self-Supervised Copy Detection (https://arxiv.org/pdf/2202.10261) for image similarity to quantify unlearning. 
- **Negative Log Likelihood (NLL)**: Likelihood of the unlearned data to quantify unlearning. Calculated according to exact likelihood formula in https://arxiv.org/pdf/2011.13456.
