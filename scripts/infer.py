# sripts/infer.py
import torch
from src.models.jsa import JSA
from torch.utils.data import DataLoader
from src.data.mnist import MNISTDataset
import math
import numpy as np
import os
import logging
import shutil

from src.utils.codebook_utils import (
    encode_multidim_to_index,
    decode_index_to_multidim,
    plot_codebook_usage_distribution,
    save_images_grid,
)




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
            multi_dim_index = decode_index_to_multidim(
                index, num_categories
            ).to(model.device)  # [num_latent_vars]
            
            # Expand to [B, H, W, num_latent_vars] for batch size 1, height 1, width 1
            multi_dim_index = multi_dim_index.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1,1,1,num_latent_vars]

            # Decode image
            decoded = model.joint_model.decode(multi_dim_index)
            decoded_images.append(decoded.cpu().squeeze(0).numpy())  # [28*28]
    return decoded_images


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
    model = JSA.load_from_checkpoint(
       checkpoint_path=checkpoint_path
    )
    model=model.to(device)
    model.eval()

    # Prepare test data
    test_dataset = MNISTDataset(root="./data", train=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    num_latent_vars = model.proposal_model.num_latent_vars
    num_categories = model.proposal_model._num_categories

    codebook_size = math.prod(num_categories)
    codebook_counter = np.zeros(codebook_size, dtype=np.int32)
    logger.info(
        f"Number of latent variables: {num_latent_vars}, Codebook size: {codebook_size}"
    )

    # Iterate over test data and count codebook usage
    with torch.no_grad():  # Disable gradient computation
        for batch in test_loader:
            x, _, idx = batch
            x = x.to(device)
            h = model.proposal_model.encode(x)  # [B, H, W, num_latent_vars]
            h = h.view(-1, num_latent_vars) # [B*H*W, num_latent_vars]
            # print("Sampled latent variables h shape:", h.shape)

            h = h.cpu().numpy()  # [B, num_latent_vars]
            indices = encode_multidim_to_index(h, num_categories)
            for index in indices:
                codebook_counter[index] += 1
    # Compute codebook utilization
    used_codewords = np.sum(codebook_counter > 0)
    utilization_rate = used_codewords / codebook_size * 100
    logger.info(f"Used codewords: {used_codewords}/{codebook_size}")
    logger.info(f"Codebook utilization rate: {utilization_rate:.4f}%")

    # Find indices of used codewords
    used_codeword_indices = np.where(codebook_counter > 0)[0]
    logger.info(f"Used codeword indices: {used_codeword_indices}")

    # Decode used codewords to inspect corresponding images
    model.joint_model.eval()
    decoded_images = decode_images(
        used_codeword_indices, model, num_categories
    )
    save_images_grid(
        decoded_images,
        save_path=f"{infer_dir}/decoded_codewords",
        images_per_page=100,
        grid_size=(10, 10),
        title="Decoded Images from Used Codewords",
        save_to_disk=True,
    )

    # Also decode unused codewords to inspect their images
    unused_codeword_indices = np.where(codebook_counter == 0)[0]
    logger.info(f"Unused codeword indices: {unused_codeword_indices}")
    # Select a subset of unused codewords to decode
    num_unused_to_decode = min(100, len(unused_codeword_indices))
    unused_codeword_indices = unused_codeword_indices[:num_unused_to_decode]

    decoded_unused_images = decode_images(
        unused_codeword_indices, model, num_categories
    )

    # Save decoded images, 100 per page in a 10x10 grid
    save_images_grid(
        decoded_unused_images,
        save_path=f"{infer_dir}/decoded_unused_codewords",
        images_per_page=100,
        grid_size=(10, 10),
        title="Decoded Images from Unused Codewords",
        save_to_disk=True,
    )

    # Plot 1D distribution
    plot_codebook_usage_distribution(
        codebook_counter,
        codebook_size,
        used_codewords,
        utilization_rate,
        save_path=f"{infer_dir}/codebook_usage_distribution.png",
        sort_by_counter=False,
        save_to_disk=True,
    )


if __name__ == "__main__":

    exp_dir = "egs/continuous_mnist/categorical_prior_conv/version_0"
    config_path = f"{exp_dir}/config.yaml"
    checkpoint_path = f"{exp_dir}/checkpoints/best-checkpoint.ckpt"
    main(exp_dir, config_path, checkpoint_path)
