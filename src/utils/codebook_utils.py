# src/utils/codebook_utils.py
import torch
import math
from typing import List, Union
import numpy as np
import matplotlib.pyplot as plt


def compute_category_weights(
    num_categories: Union[List[int], torch.Tensor], device=None, dtype=torch.long
):
    """
    Compute weight vector w such that w[i] = prod_{j=i+1..K-1} num_categories[j]
    Returns LongTensor of shape [K]
    """
    if not isinstance(num_categories, torch.Tensor):
        num_categories = torch.tensor(list(num_categories), dtype=dtype, device=device)
    else:
        num_categories = num_categories.to(device=device).to(dtype)

    K = num_categories.numel()
    if K == 0:
        return torch.tensor([], dtype=dtype, device=device)

    # We can compute by cumulative product on reversed array
    # Example: num_categories = [c0, c1, c2]
    # cumprod(reverse) = [c2, c1*c2, c0*c1*c2], flip -> [c0*c1*c2, c1*c2, c2]
    # then roll left and set last to 1
    rev_cumprod = torch.cumprod(num_categories.flip(0), dim=0).flip(0)
    weights = torch.roll(rev_cumprod, shifts=-1, dims=0)
    weights[-1] = 1
    return weights.to(torch.long)


def encode_multidim_to_index(
    multi_dim_index: torch.Tensor, num_categories: Union[List[int], torch.Tensor]
) -> torch.Tensor:
    """
    multi_dim_index: LongTensor of shape [B, K] or [K]
    num_categories: list or tensor of length K
    returns: LongTensor of shape [B] (or scalar if input was 1D)
    """
    # ensure tensor
    if not isinstance(multi_dim_index, torch.Tensor):
        multi_dim_index = torch.tensor(multi_dim_index, dtype=torch.long)

    device = multi_dim_index.device
    weights = compute_category_weights(num_categories, device=device)  # [K]

    if multi_dim_index.dim() == 1:
        # [K] -> scalar
        return (multi_dim_index.to(weights.dtype) * weights).sum().long()
    else:
        # [B, K] * [K] -> [B, K] then sum -> [B]
        return (
            (multi_dim_index.to(weights.dtype) * weights.unsqueeze(0))
            .sum(dim=-1)
            .long()
        )


def decode_index_to_multidim(
    index: torch.Tensor, num_categories: Union[List[int], torch.Tensor]
) -> torch.Tensor:
    """
    index: LongTensor of shape [B] or scalar (0-d tensor)
    num_categories: list or tensor of length K
    returns: LongTensor of shape [B, K] with values in [0, c_i-1]
    NOTE: batch-parallel on index, uses a small loop over K (usually small).
    """
    # make 1D batch
    was_scalar = False
    if not isinstance(index, torch.Tensor):
        index = torch.tensor(index, dtype=torch.long)
    if index.dim() == 0:
        index = index.unsqueeze(0)
        was_scalar = True

    device = index.device
    num_categories_t = (
        num_categories
        if isinstance(num_categories, torch.Tensor)
        else torch.tensor(list(num_categories), dtype=torch.long, device=device)
    )
    K = num_categories_t.numel()
    weights = compute_category_weights(num_categories_t, device=device)  # [K]

    # prepare output
    B = index.size(0)
    out = torch.empty((B, K), dtype=torch.long, device=device)

    remainder = index.clone()  # [B]

    for i in range(K):
        w = weights[i]  # scalar
        # floor division and remainder
        digit = remainder // w
        out[:, i] = digit
        remainder = remainder % w

    if was_scalar:
        return out.squeeze(0)  # return [K]
    return out  # [B, K]


def save_images_grid(
    images,
    save_path=None,
    tag_prefix="",
    images_per_page=100,
    grid_size=(10, 10),
    title=None,
    save_to_disk=True,
):
    """
    Save images arranged in grid pages and return a dictionary of figures.

    Args:
        images: list or array of images, each shape [H * W] or [H, W].
        save_path: path prefix (without extension). Will save _page_{i}.png if save_to_disk is True.
        tag_prefix: Prefix for figure tags in the returned dictionary.
        images_per_page: Number of images per page.
        grid_size: Grid size (rows, cols), default is (10, 10).
        title: Title for the grid.
        save_to_disk: Whether to save images to disk (default: True).

    Returns:
        A dictionary of {name: fig} where `name` is the tag and `fig` is the matplotlib figure.
    """
    num_images = len(images)
    if num_images == 0:
        return {}

    rows, cols = grid_size
    per_page = images_per_page
    num_pages = math.ceil(num_images / per_page)
    H = int(np.sqrt(images[0].size)) if images[0].ndim == 1 else images[0].shape[0]

    figures = {}  # Dictionary to store figures for external use

    for page in range(num_pages):
        fig = plt.figure(figsize=(cols, rows))
        start = page * per_page
        end = min(start + per_page, num_images)
        for i in range(start, end):
            idx = i - start
            plt.subplot(rows, cols, idx + 1)
            img = images[i]
            if img.ndim == 1:
                img = img.reshape(int(math.sqrt(img.size)), -1)
            plt.imshow(img, cmap="gray")
            plt.axis("off")
        plt.tight_layout()
        if title:
            plt.suptitle(title, fontsize=12)

        # Generate tag for this page
        tag = f"{tag_prefix}_page_{page + 1}" if tag_prefix else f"page_{page + 1}"
        figures[tag] = fig

        # Save to disk if required
        if save_to_disk and save_path:
            out_path = f"{save_path}_page_{page + 1}.png"
            plt.savefig(out_path)

        # Close the figure to avoid memory leaks
        plt.close()

    return figures


def plot_codebook_usage_distribution(
    codebook_counter,
    codebook_size,
    used_codewords,
    utilization_rate,
    tag_prefix="",
    save_path=None,
    sort_by_counter=False,
    save_to_disk=True,
):
    """
    Plot codebook usage distribution and return a dictionary of figures.

    Args:
        codebook_counter: Counter array recording usage count for each codeword.
        codebook_size: Total size of the codebook.
        used_codewords: Number of codewords that were used.
        utilization_rate: Codebook utilization rate (percentage).
        tag_prefix: Prefix for figure tags in the returned dictionary.
        save_path: Path to save the plot (if save_to_disk is True).
        sort_by_counter: Whether to sort by count in descending order.
        save_to_disk: Whether to save the plot to disk (default: True).

    Returns:
        A dictionary of {name: fig} where `name` is the tag and `fig` is the matplotlib figure.
    """
    figures = {}  # Dictionary to store figures for external use

    # Create the figure
    fig = plt.figure(figsize=(12, 6))

    if sort_by_counter:
        # Sort by counter descending
        sorted_indices = np.argsort(codebook_counter)[::-1]
        sorted_counter = codebook_counter[sorted_indices]
        plt.bar(range(codebook_size), sorted_counter, width=1.0)
    else:
        # Show in default order
        plt.bar(range(codebook_size), codebook_counter, width=1.0)

    plt.xlabel("Codeword Index")
    plt.ylabel("Usage Count")
    plt.text(
        codebook_size * 0.7,
        max(codebook_counter) * 0.9,
        f"Used codewords: {used_codewords}/{codebook_size}\nUtilization rate: {utilization_rate:.4f}%",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.5),
    )
    plt.title("Codebook Usage Distribution")
    plt.tight_layout()

    # Generate tag for the figure
    tag = (
        f"{tag_prefix}_codebook_usage_distribution"
        if tag_prefix
        else "codebook_usage_distribution"
    )
    figures[tag] = fig

    # Save to disk if required
    if save_to_disk and save_path:
        plt.savefig(save_path)

    # Close the figure to avoid memory leaks
    plt.close()

    return figures


if __name__ == "__main__":
    # create categories
    num_categories = [4, 5, 6]  # K = 3, codebook size = 120

    # encode-decode roundtrip test
    multi = torch.tensor(
        [[0, 0, 0], [3, 4, 5], [2, 1, 3], [0, 2, 1]], dtype=torch.long
    )  # [4,3]
    idx = encode_multidim_to_index(multi, num_categories)  # [4]
    print("idx:", idx)

    # decode back
    multi2 = decode_index_to_multidim(idx, num_categories)
    print("multi2:", multi2)
    print("equal:", torch.all(multi == multi2))

    # scalar test
    i = torch.tensor(37, dtype=torch.long)
    m = decode_index_to_multidim(i, num_categories)  # [K]
    i2 = encode_multidim_to_index(m, num_categories)
    print("scalar roundtrip:", i.item(), i2.item())
