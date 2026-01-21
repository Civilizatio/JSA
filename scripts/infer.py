# sripts/infer.py
import torch
from src.models.jsa import JSA
from src.models.vq_gan import VQModel
from torch.utils.data import DataLoader
from src.data.mnist import MNISTDataset
from src.data.cifar10 import CIFAR10Dataset
import math
import numpy as np
import os
import logging
import shutil
from lightning.pytorch.cli import LightningCLI
import torch.nn.functional as F

from src.utils.codebook_utils import (
    encode_multidim_to_index,
    decode_index_to_multidim,
    plot_codebook_usage_distribution,
    save_images_grid,
)

from tqdm import tqdm

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPSMetric
from torchmetrics.image.fid import FrechetInceptionDistance as FIDMetric
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure as SSIMMetric


class InferenceCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # Allow these keys in config by adding them as arguments
        parser.add_argument("--ckpt_path", type=str, default=None)


def decode_images(indices, model, num_categories):
    """Decode given codeword indices into images.

    Args:
        indices: 1D numpy array of codeword indices.
        model: Trained JSA model.
        num_categories: Number of categories for each latent variable.
    """
    decoded_images = []
    with torch.no_grad():
        for index in indices:
            # Convert 1D index to multi-dimensional category indices
            multi_dim_index = decode_index_to_multidim(index, num_categories).to(
                model.device
            )  # [num_latent_vars]

            # Expand to [B, H, W, num_latent_vars] for batch size 1, height 1, width 1
            multi_dim_index = (
                multi_dim_index.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            )  # [1,1,1,num_latent_vars]

            # Decode image
            decoded = model.joint_model.decode(multi_dim_index)
            decoded_images.append(decoded.cpu().squeeze(0).numpy())  # [28*28]
    return decoded_images


def get_model_info(model):
    """Get model type and codebook information."""
    if isinstance(model, VQModel):
        model_type = "VQModel"
        codebook_size = model.get_codebook_size()
        num_categories = [codebook_size]
        num_latent_vars = 1
    elif isinstance(model, JSA):
        model_type = "JSA"
        num_latent_vars = model.proposal_model.num_latent_vars
        num_categories = model.proposal_model._num_categories
        codebook_size = math.prod(num_categories)
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
    return model_type, num_latent_vars, num_categories, codebook_size


def inference_step(batch, model, device):
    # Move input to device
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }
    # Get codebook indices

    indices = model.get_codebook_indices(batch)
    x = model.get_input(batch, model.dataset_key["image_key"])
    if isinstance(batch, dict) and "label" in batch:
        y = batch["label"]
    elif isinstance(batch, (list, tuple)) and len(batch) > 1:
        y = batch[1]
    else:
        y = None

    # Reconstruct images
    if isinstance(model, VQModel):
        x_hat, _ = model.forward(x)
    elif isinstance(model, JSA):
        x_hat = model.forward(x)

    return x, x_hat, y, indices


# ================= Main Inference Script ================= #


def main(exp_dir, config_path, checkpoint_path):
    # Load model
    infer_dir = f"{exp_dir}/inference"
    if os.path.exists(infer_dir):
        shutil.rmtree(infer_dir)
    os.makedirs(infer_dir, exist_ok=True)

    logging.basicConfig(
        filename=f"{infer_dir}/inference.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info("Starting inference...")
    logger.info(f"Loading model from {checkpoint_path}...")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cli = InferenceCLI(
        run=False,
        args=[
            "--config",
            config_path,
        ],
    )

    model = cli.model
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()

    # Prepare test data
    test_dataset = CIFAR10Dataset(root="./data/cifar10", train=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Get model info
    model_type, num_latent_vars, num_categories, codebook_size = get_model_info(model)
    logger.info(f"Model type: {model_type}")
    logger.info(f"Number of latent variables: {num_latent_vars}")
    logger.info(f"Number of categories per latent variable: {num_categories}")
    logger.info(f"Total codebook size: {codebook_size}")

    codebook_counter = torch.zeros(codebook_size, dtype=torch.long, device=device)

    # Prepare metrics
    metrics = {
        "lpips": LPIPSMetric(net_type="vgg").to(device),
        "fid": FIDMetric(feature=2048).to(device),
        "ssim": SSIMMetric(data_range=1.0).to(device),
    }
    stats = {
        "mse": 0.0,
        "l1": 0.0,
        "total_samples": 0,
    }
    vis_samples = dict()  # To store visualization samples per class

    # Iterate over test data and count codebook usage
    with torch.no_grad():  # Disable gradient computation
        for batch in tqdm(test_loader):

            x, x_hat, y, indices = inference_step(batch, model, device)

            # Get codebook indices and update counter
            codebook_counter.index_add_(
                0,
                indices,
                torch.ones_like(indices, dtype=torch.long, device=device),
            )

            # Update metrics
            # MSE/L1
            stats["mse"] += F.mse_loss(x_hat, x, reduction="sum").item()
            stats["l1"] += F.l1_loss(x_hat, x, reduction="sum").item()
            stats["total_samples"] += x.size(0)

            # Advance metrics
            # LPIPS: [-1, 1]
            x_lpips = x.clamp(-1, 1)
            x_hat_lpips = x_hat.clamp(-1, 1)
            metrics["lpips"].update(x_lpips, x_hat_lpips)

            # SSIM: [0, 1]
            x_ssim = torch.clamp((x + 1.0) / 2.0, 0, 1)
            x_hat_ssim = torch.clamp((x_hat + 1.0) / 2.0, 0, 1)
            metrics["ssim"].update(x_ssim, x_hat_ssim)

            # FID: 0~255, UINT8
            x_fid = ((x + 1.0) / 2.0 * 255).to(torch.uint8)
            x_hat_fid = ((x_hat + 1.0) / 2.0 * 255).to(torch.uint8)
            metrics["fid"].update(x_fid, real=True)
            metrics["fid"].update(x_hat_fid, real=False)

            if len(vis_samples) < 10:
                batch_y = y.cpu().numpy()
                x_cpu = x.cpu().numpy()
                x_hat_cpu = x_hat.cpu().numpy()
                for i in range(x.size(0)):
                    if batch_y[i] not in vis_samples:
                        vis_samples[batch_y[i]] = (x_cpu[i], x_hat_cpu[i])
                    if len(vis_samples) >= 10:
                        break

    # Compute codebook utilization
    used_codewords = torch.sum(codebook_counter > 0).item()
    utilization_rate = used_codewords / codebook_size * 100
    logger.info(f"Used codewords: {used_codewords}/{codebook_size}")
    logger.info(f"Codebook utilization rate: {utilization_rate:.4f}%")

    avg_mse = stats["mse"] / stats["total_samples"]
    avg_l1 = stats["l1"] / stats["total_samples"]
    final_lpips = metrics["lpips"].compute().item()
    final_fid = metrics["fid"].compute().item()
    final_ssim = metrics["ssim"].compute().item()
    logger.info(f"Average MSE on test set: {avg_mse:.6f}")
    logger.info(f"Average L1 Loss on test set: {avg_l1:.6f}")
    logger.info(f"Average LPIPS on test set: {final_lpips:.6f}")
    logger.info(f"FID between test set and reconstructions: {final_fid:.6f}")
    logger.info(f"Average SSIM on test set: {final_ssim:.6f}")

    if len(vis_samples) > 0:
        sorted_keys = sorted(vis_samples.keys())
        orig_images = torch.stack(
            [torch.tensor(vis_samples[k][0]) for k in sorted_keys]
        )
        recon_images = torch.stack(
            [torch.tensor(vis_samples[k][1]) for k in sorted_keys]
        )
        save_images_grid(
            images=orig_images,
            save_path=f"{infer_dir}/original_images.png",
            nrow=len(orig_images),
        )
        save_images_grid(
            images=recon_images,
            save_path=f"{infer_dir}/reconstructed_images.png",
            nrow=len(recon_images),
        )
        comparison = torch.cat([orig_images, recon_images], dim=0)
        save_images_grid(
            images=comparison,
            save_path=f"{infer_dir}/images_comparison.png",
            nrow=len(orig_images),
        )

    # Plot 1D distribution
    _, code_entropy = plot_codebook_usage_distribution(
        codebook_counter.cpu().numpy(),
        codebook_size,
        save_path=f"{infer_dir}/codebook_usage_distribution.png",
        sort_by_counter=True,
        save_to_disk=True,
        use_log_scale=False,
    )
    logger.info(f"Codebook utilization entropy: {code_entropy:.4f} bits")

    # Clear loggers
    if logger.hasHandlers():
        logger.handlers.clear()
    

if __name__ == "__main__":

    dir_list = [
       "egs/cifar10/jsa/categorical_prior_conv/2026-01-20_15-49-05",
    ]
    
    for exp_dir in dir_list:
        config_path = f"{exp_dir}/config.yaml"
        checkpoint_path = f"{exp_dir}/checkpoints/last.ckpt"
        main(exp_dir, config_path, checkpoint_path)