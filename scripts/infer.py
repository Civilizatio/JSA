# sripts/infer.py
import torch
from src.models.jsa import JSA
from src.models.vq_gan import VQModel
from torch.utils.data import DataLoader, Subset
from src.data.mnist import MNISTDataset
from src.data.cifar10 import CIFAR10Dataset
import math
import numpy as np
import os
import logging
import shutil
from lightning.pytorch.cli import LightningCLI
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.utils.codebook_utils import (
    encode_multidim_to_index,
    decode_index_to_multidim,
    plot_codebook_usage_distribution,
    save_images_grid,
)
from src.utils.distribution_analysis import (
    normalize_counter,
    js_divergence_matrix,
    topk_overlap_matrix,
    plot_heatmap,
    pca_embedding,
    umap_embedding,
    plot_embedding,
)


from tqdm import tqdm

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPSMetric
from torchmetrics.image.fid import FrechetInceptionDistance as FIDMetric
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure as SSIMMetric


# ================ Utility Classes and Functions ================ #
class InferenceModule:
    """Base interface for inference modules.

    Three stages: setup, update, finalize.
    """

    def setup(self, device, logger, save_dir):
        pass

    def update(self, batch, outputs):
        pass

    def finalize(self):
        pass


class MetricTracker(InferenceModule):
    """Handles all numerical metrics during inference.

    Metrics include:
    - MSE
    - L1
    - LPIPS
    - FID
    - SSIM
    """

    def __init__(self):
        self.metrics = dict()
        self.stats = {
            "mse": 0.0,
            "l1": 0.0,
            "total_samples": 0,
        }

    def setup(self, device, logger, save_dir):
        self.device = device
        self.logger = logger
        self.metrics = {
            "lpips": LPIPSMetric(net_type="vgg").to(device),
            "fid": FIDMetric(feature=2048).to(device),
            "ssim": SSIMMetric(data_range=1.0).to(device),
        }

    def update(self, batch, outputs):
        x = outputs["x"]
        x_hat = outputs["x_hat"]

        # Basic stats
        self.stats["mse"] += F.mse_loss(x_hat, x, reduction="sum").item()
        self.stats["l1"] += F.l1_loss(x_hat, x, reduction="sum").item()
        self.stats["total_samples"] += x.size(0)

        # Advance metrics
        # LPIPS: [-1, 1]
        x_lpips = x.clamp(-1, 1)
        x_hat_lpips = x_hat.clamp(-1, 1)
        self.metrics["lpips"].update(x_lpips, x_hat_lpips)

        # SSIM: [0, 1]
        x_ssim = torch.clamp((x + 1.0) / 2.0, 0, 1)
        x_hat_ssim = torch.clamp((x_hat + 1.0) / 2.0, 0, 1)
        self.metrics["ssim"].update(x_ssim, x_hat_ssim)

        # FID: 0~255, UINT8
        x_fid = ((x + 1.0) / 2.0 * 255).to(torch.uint8)
        x_hat_fid = ((x_hat + 1.0) / 2.0 * 255).to(torch.uint8)
        self.metrics["fid"].update(x_fid, real=True)
        self.metrics["fid"].update(x_hat_fid, real=False)

    def finalize(self):
        avg_mse = self.stats["mse"] / self.stats["total_samples"]
        avg_l1 = self.stats["l1"] / self.stats["total_samples"]
        final_lpips = self.metrics["lpips"].compute().item()
        final_fid = self.metrics["fid"].compute().item()
        final_ssim = self.metrics["ssim"].compute().item()

        self.logger.info(f"Average MSE on test set: {avg_mse:.6f}")
        self.logger.info(f"Average L1 Loss on test set: {avg_l1:.6f}")
        self.logger.info(f"Average LPIPS on test set: {final_lpips:.6f}")
        self.logger.info(f"FID between test set and reconstructions: {final_fid:.6f}")
        self.logger.info(f"Average SSIM on test set: {final_ssim:.6f}")


class CodebookUsageTracker(InferenceModule):
    """Tracks codebook usage statistics during inference.

    Includes:
    - Global codeword usage count
    - Per-Class codeword usage (if labels available)
    - Per-Spatial Position codeword usage (if spatial shape provided)
    """

    def __init__(
        self, codebook_size, class_names=None, track_per_class=False, spatial_shape=None
    ):
        self.codebook_size = codebook_size
        self.class_names = class_names if class_names else []
        self.track_per_class = track_per_class
        self.spatial_shape = spatial_shape  # Tuple (H, W) or None

    def setup(self, device, logger, save_dir):
        self.device = device
        self.logger = logger
        self.save_dir = save_dir

        # Global codebook counter
        self.codebook_counter = torch.zeros(
            self.codebook_size, dtype=torch.long, device=device
        )

        # Per-class codebook counters
        if self.track_per_class and len(self.class_names) > 0:
            self.class_counters = torch.zeros(
                (len(self.class_names), self.codebook_size),
                dtype=torch.long,
                device=device,
            )

        # Per-spatial position codebook counters
        if self.spatial_shape is not None:
            self.num_spatial_positions = self.spatial_shape[0] * self.spatial_shape[1]
            self.spatial_counters = torch.zeros(
                (self.num_spatial_positions, self.codebook_size),
                dtype=torch.long,
                device=device,
            )

    def set_model(self, model):
        self.model = model

    def update(self, batch, outputs):
        indices = outputs["indices"]  # [B*H*W*num_latent_vars, ]
        y = outputs.get("y", None)  # [B, ] or None

        # Update global counter
        self.codebook_counter.index_add_(
            0,
            indices,
            torch.ones_like(indices, dtype=torch.long, device=self.device),
        )

        # Update per-class counters
        if self.track_per_class and y is not None:
            # Interleaving y to match indices shape if needed
            y = (
                y.unsqueeze(1).expand(-1, indices.size(0) // y.size(0)).reshape(-1)
            )  # [B*H*W*num_latent_vars, ]

            for class_idx, class_name in enumerate(self.class_names):
                mask = y == class_idx
                if mask.any():
                    selected_indices = indices[mask]
                    self.class_counters[class_idx].index_add_(
                        0,
                        selected_indices,
                        torch.ones_like(
                            selected_indices, dtype=torch.long, device=self.device
                        ),
                    )

        # Update per-spatial position counters
        if self.spatial_shape is not None:
            # indices shape: [B*H*W*num_latent_vars, ]
            # we need to separate B from H*W*num_latent_vars
            # Here we assume num_latent_vars is 1 for simplicity
            B = batch["image"].size(0)
            indices_spatial = indices.view(B, self.num_spatial_positions)  # [B, H*W]
            for pos in range(self.num_spatial_positions):
                pos_indices = indices_spatial[:, pos]
                self.spatial_counters[pos].index_add_(
                    0,
                    pos_indices,
                    torch.ones_like(pos_indices, dtype=torch.long, device=self.device),
                )

    def analyze_distributions(
        self,
        counters: torch.Tensor,
        name: str,
        save_dir: str,
        topk: int = 50,
        labels=None,
        names=None,
        annotate=False,
    ):
        """
        counters: [N, codebook_size]
        """
        os.makedirs(save_dir, exist_ok=True)

        # ---- normalize ----
        P = np.stack([normalize_counter(c.cpu().numpy()) for c in counters])  # [N, K]

        # ---- JS divergence ----
        js_mat = js_divergence_matrix(P)
        plot_heatmap(
            js_mat,
            title=f"{name} JS Divergence",
            save_path=f"{save_dir}/{name}_js_divergence.png",
        )

        # ---- Top-K overlap ----
        overlap_mat = topk_overlap_matrix(P, k=topk)
        plot_heatmap(
            overlap_mat,
            title=f"{name} Top-{topk} Overlap",
            save_path=f"{save_dir}/{name}_top{topk}_overlap.png",
            cmap="magma",
        )

        # ---- PCA ----
        emb_pca, var_ratio = pca_embedding(P)
        plot_embedding(
            emb=emb_pca,
            labels=labels,
            names=names,
            annotate=annotate,
            title=f"{name} PCA (var={var_ratio[0]:.2f},{var_ratio[1]:.2f})",
            save_path=f"{save_dir}/{name}_pca.png",
        )

        # ---- UMAP (optional) ----
        try:
            emb_umap = umap_embedding(P)
            plot_embedding(
                emb=emb_umap,
                labels=labels,
                names=names,
                annotate=annotate,
                title=f"{name} UMAP",
                save_path=f"{save_dir}/{name}_umap.png",
            )
        except Exception as e:
            self.logger.warning(f"UMAP skipped: {e}")

    def finalize(self):

        # Global codebook usage
        used_codewords = torch.sum(self.codebook_counter > 0).item()
        utilization_rate = used_codewords / self.codebook_size * 100
        self.logger.info(f"Used codewords: {used_codewords}/{self.codebook_size}")
        self.logger.info(f"Codebook utilization rate: {utilization_rate:.4f}%")

        # Global distribution plot
        _, code_entropy = plot_codebook_usage_distribution(
            self.codebook_counter.cpu().numpy(),
            self.codebook_size,
            save_path=f"{self.save_dir}/codebook_usage_distribution.png",
            sort_by_counter=True,
            save_to_disk=True,
            use_log_scale=False,
        )
        self.logger.info(f"Codebook utilization entropy: {code_entropy:.4f} bits")

        # Per-class codebook usage
        if self.track_per_class and len(self.class_names) > 0:
            class_dist_dir = os.path.join(self.save_dir, "class_codebook_distributions")
            os.makedirs(class_dist_dir, exist_ok=True)
            self.logger.info(
                f"Saving per-class codebook distributions to {class_dist_dir}..."
            )
            for class_idx, class_name in enumerate(self.class_names):
                counter = self.class_counters[class_idx]
                _, class_entropy = plot_codebook_usage_distribution(
                    counter.cpu().numpy(),
                    self.codebook_size,
                    save_path=f"{class_dist_dir}/{class_name}_codebook_distribution.png",
                    sort_by_counter=False,
                    save_to_disk=True,
                    use_log_scale=False,
                )
                self.logger.info(
                    f"Class '{class_name}' codebook utilization entropy: {class_entropy:.4f} bits"
                )
            class_dist_dir = os.path.join(self.save_dir, "class_codebook_analysis")
            self.analyze_distributions(
                counters=self.class_counters,
                name="Class",
                save_dir=class_dist_dir,
                topk=50,
                labels=list(range(len(self.class_names))),
                names=self.class_names,
                annotate=True,
            )

        # Per-spatial position codebook usage
        if self.spatial_shape is not None:
            spatial_dist_dir = os.path.join(self.save_dir, "spatial_codebook_analysis")
            H, W = self.spatial_shape

            # Scheme A: Generate labels and names by rows
            labels = [i // W for i in range(self.num_spatial_positions)]
            names = [f"({i//W},{i%W})" for i in range(H * W)]

            # # Scheme B: Generate labels and names by radius
            # cy, cx = (H-1)/2, (W-1)/2
            # labels = []
            # names = []
            # for i in range(self.num_spatial_positions):
            #     y, x = divmod(i, W)
            #     radius = int(round(math.sqrt((y - cy) ** 2 + (x - cx) ** 2)))
            #     labels.append(radius)
            #     names.append(f"({y},{x})")

            os.makedirs(spatial_dist_dir, exist_ok=True)
            self.analyze_distributions(
                counters=self.spatial_counters,
                name="spatial",
                save_dir=spatial_dist_dir,
                topk=50,
                labels=labels,
                names=names,
                annotate=False,
            )

            # Top-K Random Reconstrunction
            topk = 10
            print_topk_per_position(
                self.spatial_counters,
                H,
                W,
                k=topk,
                logger=self.logger,
            )

            latent = sample_latent_from_topk(
                self.spatial_counters,
                H,
                W,
                k=topk,
                mode="weighted",
                device=self.device,
            )

            recon_path = (
                f"{self.save_dir}/reconstruction_from_top{topk}_per_position.png"
            )
            reconstruct_from_latent(
                model=self.model,
                latent_indices=latent,
                save_path=recon_path,
            )


class Visualizer(InferenceModule):
    """Saves a grid of original and reconstructed images during inference."""

    def __init__(self, max_samples=10):
        self.max_samples = max_samples
        self.vis_samples = dict()  # To store visualization samples per class

    def setup(self, device, logger, save_dir):
        self.logger = logger
        self.save_dir = save_dir

    def update(self, batch, outputs):
        if len(self.vis_samples) >= self.max_samples:
            return

        x = outputs["x"]
        x_hat = outputs["x_hat"]
        y = outputs.get("y", None)

        batch_y = y.cpu().numpy() if y is not None else [None] * x.size(0)
        x_cpu = x.cpu().numpy()
        x_hat_cpu = x_hat.cpu().numpy()

        for i in range(x.size(0)):
            label = batch_y[i]
            if label not in self.vis_samples:
                self.vis_samples[label] = (x_cpu[i], x_hat_cpu[i])
            if len(self.vis_samples) >= self.max_samples:
                break

    def finalize(self):
        if not self.vis_samples:
            self.logger.info("No samples collected for visualization.")
            return
        sorted_keys = sorted(self.vis_samples.keys())
        orig_images = torch.stack(
            [torch.tensor(self.vis_samples[k][0]) for k in sorted_keys]
        )
        recon_images = torch.stack(
            [torch.tensor(self.vis_samples[k][1]) for k in sorted_keys]
        )
        save_images_grid(
            images=orig_images,
            save_path=f"{self.save_dir}/original_images.png",
            nrow=len(orig_images),
        )
        save_images_grid(
            images=recon_images,
            save_path=f"{self.save_dir}/reconstructed_images.png",
            nrow=len(recon_images),
        )
        comparison = torch.cat([orig_images, recon_images], dim=0)
        save_images_grid(
            images=comparison,
            save_path=f"{self.save_dir}/images_comparison.png",
            nrow=len(orig_images),
        )
        self.logger.info(f"Saved visualization images to {self.save_dir}")


class InferenceCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # Allow these keys in config by adding them as arguments
        parser.add_argument("--ckpt_path", type=str, default=None)


def prepare_modules(run_config, codebook_size, class_names):
    """Prepare inference modules based on run configuration."""
    modules = []

    # Metric Tracker
    if run_config.get("metrics", True):
        metric_tracker = MetricTracker()
        modules.append(metric_tracker)

    # Codebook Usage Tracker
    if run_config.get("codebook_global", True) or run_config.get(
        "codebook_per_class", False
    ):
        track_per_class = run_config.get("codebook_per_class", False)
        track_spatial_shape = run_config.get("codebook_spatial_shape", None)
        codebook_tracker = CodebookUsageTracker(
            codebook_size=codebook_size,
            class_names=class_names,
            track_per_class=track_per_class,
            spatial_shape=track_spatial_shape,  # Example spatial shape, modify as needed
        )
        modules.append(codebook_tracker)

    # Visualizer
    if run_config.get("visualization", True):
        visualizer = Visualizer(max_samples=10)
        modules.append(visualizer)

    return modules


class RelabeledSubset(torch.utils.data.Dataset):
    def __init__(self, subset, label_mapping):
        self.subset = subset
        self.label_mapping = label_mapping  # old_label -> new_label

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        sample = self.subset[idx]

        if isinstance(sample, dict):
            sample = dict(sample)
            old_label = sample["label"]
            sample["label"] = self.label_mapping[int(old_label)]
            return sample

        elif isinstance(sample, (tuple, list)):
            x, y = sample
            return x, self.label_mapping[int(y)]

        else:
            raise TypeError("Unsupported sample format")


def filter_dataset_by_class(dataset, target_class_names, class_list, logger=None):
    """Filter dataset to only include samples of the target class.

    Args:
        dataset: Dataset object with 'label' attribute.
        target_class_name: Name of the target class to filter.
        class_list: List of all class names in the dataset.
    Returns:
        Filtered Subset of the dataset.
    """
    if isinstance(target_class_names, str):
        target_class_names = [target_class_names]

    target_indices = []
    for name in target_class_names:
        if name not in class_list:
            raise ValueError(f"Class name '{name}' not found in class list.")
        target_indices.append(class_list.index(name))
    old_to_new = {old: new for new, old in enumerate(target_indices)}
    target_indices_set = set(target_indices)
    indices = []

    if hasattr(dataset, "ds") and hasattr(dataset.ds, "targets"):
        # For datasets like CIFAR10Dataset
        targets = dataset.ds.targets
        indices = [i for i, label in enumerate(targets) if label in target_indices_set]
    elif hasattr(dataset, "targets"):
        print(
            f"Fast filtering using dataset.targets for class '{target_class_names}'..."
        )
        indices = [i for i, t in enumerate(dataset.targets) if t in target_indices_set]
    else:
        # For other datasets, iterate through samples
        for i in tqdm(range(len(dataset)), desc=f"Filtering {target_class_names}"):
            sample = dataset[i]
            label = None

            # For datasets returning dict samples
            if isinstance(sample, dict) and "label" in sample:
                label = sample["label"]
            # For datasets returning tuple samples (image, label)
            elif isinstance(sample, (tuple, list)) and len(sample) > 1:
                label = sample[1]

            if label in target_indices_set:
                indices.append(i)

    if logger is not None:
        logger.info(
            f"Filtered {len(indices)} samples for class '{target_class_names}'."
        )
    filtered_subset = Subset(dataset, indices)
    relabeled_subset = RelabeledSubset(filtered_subset, old_to_new)
    return relabeled_subset


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

    # Attention !
    # indices: [B*H*W*num_latent_vars, ]
    # x: [B, C, H, W]
    # x_hat: [B, C, H, W]
    # y: [B, ] or None
    return {"x": x, "x_hat": x_hat, "y": y, "indices": indices}


def print_topk_per_position(
    spatial_counters: torch.Tensor,
    H: int,
    W: int,
    k: int = 10,
    logger=None,
):
    """
    spatial_counters: [H*W, codebook_size]
    """
    counters = spatial_counters.cpu().numpy()

    logger.info("=" * 80)
    logger.info(f"Top-{k} codewords per spatial position")
    logger.info("=" * 80)

    for pos in range(H * W):
        row = pos // W
        col = pos % W

        freq = counters[pos]
        prob = freq / (freq.sum() + 1e-12)

        topk_idx = np.argsort(prob)[-k:][::-1]
        topk_prob = prob[topk_idx]

        header = f"[pos=({row},{col})]"
        logger.info(header)
        for i, (idx, p) in enumerate(zip(topk_idx, topk_prob)):
            logger.info(f"  {i:02d}: idx={idx:5d}, freq={freq[idx]:8d}, prob={p:.4f}")
        logger.info("-" * 60)

    if logger is not None:
        logger.info(f"Printed Top-{k} statistics for all spatial positions.")


def sample_latent_from_topk(
    spatial_counters: torch.Tensor,
    H: int,
    W: int,
    k: int = 10,
    mode: str = "weighted",  # "uniform" | "weighted"
    device="cuda",
):
    """
    Return:
        latent_indices: LongTensor [1, H, W, 1]
    """
    counters = spatial_counters.cpu().numpy()
    sampled = []

    for pos in range(H * W):
        freq = counters[pos]
        prob = freq / (freq.sum() + 1e-12)

        topk_idx = np.argsort(prob)[-k:]

        if mode == "uniform":
            choice = np.random.choice(topk_idx)
        elif mode == "weighted":
            topk_prob = prob[topk_idx]
            topk_prob = topk_prob / topk_prob.sum()
            choice = np.random.choice(topk_idx, p=topk_prob)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        sampled.append(choice)

    sampled = np.array(sampled).reshape(1, H, W, 1)
    return torch.from_numpy(sampled).long().to(device)


def reconstruct_from_latent(
    model,
    latent_indices: torch.Tensor,
    save_path: str,
):
    """
    latent_indices: [1, H, W, num_latent_vars]
    """
    with torch.no_grad():
        if isinstance(model, JSA):
            x_hat = model.decode(latent_indices)
        elif isinstance(model, VQModel):
            x_hat = model.decode_code(latent_indices.squeeze(-1))

    save_images_grid(
        images=x_hat.cpu(),
        save_path=save_path,
        nrow=1,
    )


# ================= Main Inference Script ================= #


def main(exp_dir, config_path, checkpoint_path, run_config=None):
    """
    Args:
        run_config: dict, Configuration dictionary to enable/disable modules
    """

    # Default run configuration
    if run_config is None:
        run_config = {
            "metrics": True,
            "visualization": True,
            "codebook_global": True,
            "codebook_per_class": False,
            "target_class_names": None,
        }

    # Setup logging and directories
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

    # Load model
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
    test_dataset = CIFAR10Dataset(root="./data/cifar10", train=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    CIFAR10_CLASSES = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    target_class_names = run_config.get("target_class_names", None)
    if target_class_names is not None:
        logger.info(f"Filtering test dataset for classes: {target_class_names}...")
        test_dataset = filter_dataset_by_class(
            test_dataset, target_class_names, CIFAR10_CLASSES, logger=logger
        )
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Get model info
    model_type, num_latent_vars, num_categories, codebook_size = get_model_info(model)
    logger.info(f"Model type: {model_type}")
    logger.info(f"Number of latent variables: {num_latent_vars}")
    logger.info(f"Number of categories per latent variable: {num_categories}")
    logger.info(f"Total codebook size: {codebook_size}")

    active_modules = prepare_modules(
        run_config,
        codebook_size,
        class_names=target_class_names if target_class_names else CIFAR10_CLASSES,
    )
    logger.info(
        f"Active inference modules: {[type(m).__name__ for m in active_modules]}"
    )
    for module in active_modules:
        module.setup(device, logger, infer_dir)
        if hasattr(module, "set_model"):
            module.set_model(model)

    # ========== Main Inference Loop ========== #
    with torch.no_grad():  # Disable gradient computation
        model.sampler.to(device)
        for batch in tqdm(test_loader):

            # outputs = inference_step(batch, model, device)
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            x = model.get_input(batch, model.dataset_key["image_key"])
            idx = model.get_input(batch, model.dataset_key["index_key"])
            logits = model.proposal_model(x)[0]
            probs = torch.softmax(logits, dim=-1)
            logger.info(f"Probs shape: {probs.shape}. Showing Top-3 concentration pre positions:")
            
            sample_probs=probs[0]
            H, W, num_latent_vars = sample_probs.shape

            topk_probs, topk_indices = torch.topk(sample_probs, k=3, dim=-1)
            for h in range(H):
                for w in range(W):
                    
                    vals = topk_probs[h, w].tolist()
                    inds = topk_indices[h, w].tolist()
                    
                    top3_str=", ".join([f"{idx}: {p:.4f}" for idx, p in zip(inds, vals)])
                    logger.info(f"Position ({h},{w}) Top-3: {top3_str}")
            
            h = model.sampler.sample(
                x, idx=idx, num_steps=4, parallel=False, return_all=True
            )

            logger.info(f"Sampled latent h with shape: {h.shape}")
            logger.info(f"h: {h}")
            logger.info(f"h min/max: {h.min().item()}/{h.max().item()}")
            # Update all modules
            # for module in active_modules:
            #     module.update(batch, outputs)

    # ========== Finalization ========== #
    logger.info("Finalizing inference modules...")
    for module in active_modules:
        module.finalize()

    # Clear loggers
    if logger.hasHandlers():
        logger.handlers.clear()


if __name__ == "__main__":

    dir_list = [
        # "egs/cifar10/jsa/categorical_prior_conv/2026-01-15_15-41-30",
        "egs/cifar10/jsa/categorical_prior_conv/2026-01-26_00-06-47",
    ]
    target_class_names = None

    run_config = {
        "metrics": False,
        "visualization": False,
        "codebook_global": False,
        "codebook_per_class": False,
        "codebook_spatial_shape": None,  # Example spatial shape
        "target_class_names": target_class_names,
    }

    for exp_dir in dir_list:
        config_path = f"{exp_dir}/config.yaml"
        checkpoint_path = f"{exp_dir}/checkpoints/last.ckpt"
        main(exp_dir, config_path, checkpoint_path, run_config=run_config)
