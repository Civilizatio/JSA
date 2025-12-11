# src/models/jsa.py
import torch
from lightning.pytorch import LightningModule
from hydra.utils import instantiate
import torchvision
import torch.distributed as dist

from src.samplers.misampler import MISampler
from src.base.base_jsa_modules import BaseJointModel, BaseProposalModel
from utils.codebook_utils import (
    encode_multidim_to_index,
    decode_index_to_multidim,
    compute_category_weights,
    plot_codebook_usage_distribution,
    save_images_grid
)
import math
import numpy as np
import matplotlib.pyplot as plt


class JSA(LightningModule):
    def __init__(
        self,
        joint_model,
        proposal_model,
        sampler,
        lr_joint=1e-3,
        lr_proposal=1e-3,
        num_mis_steps=3,
        cache_start_epoch=0,
    ):
        super().__init__()
        self.joint_model: BaseJointModel = instantiate(joint_model)
        self.proposal_model: BaseProposalModel = instantiate(proposal_model)
        self.sampler: MISampler = instantiate(
            sampler,
            joint_model=self.joint_model,
            proposal_model=self.proposal_model,
        )

        self.automatic_optimization = False
        self.lr_joint = lr_joint
        self.lr_proposal = lr_proposal
        self.num_mis_steps = num_mis_steps
        self.cache_start_epoch = cache_start_epoch

        # For visualization during validation
        self.validation_step_outputs = []

        self.log_codebook_utilization_valid = False
        self.log_codebook_utilization_test = False

    def setup(self, stage=None):
        device = self.device
        self.sampler.to(device)

    def forward(self, x, idx=None):

        if idx is not None:
            h = self.sampler.sample(x, idx=idx, num_steps=self.num_mis_steps)
        else:  # use proposal model directly
            h = self.proposal_model.sample_latent(
                x, num_samples=1
            )  # [B, latent_dim, 1]
            h = h.squeeze(-1)  # [B, latent_dim]
        x_hat = self.joint_model.sample(h=h)
        return x_hat

    def configure_optimizers(self):
        opt_joint = torch.optim.Adam(self.joint_model.parameters(), lr=self.lr_joint)
        opt_proposal = torch.optim.Adam(
            self.proposal_model.parameters(), lr=self.lr_proposal
        )
        return [opt_joint, opt_proposal]

    # ========================= Training =========================

    def on_train_epoch_start(self):
        if self.current_epoch >= self.cache_start_epoch:
            self.sampler.use_cache = True
        else:
            self.sampler.use_cache = False

    def training_step(self, batch, batch_idx):
        x, _, idx = batch  # x: [B, D], idx: [B,]

        # MISampling step
        h = self.sampler.sample(x, idx=idx, num_steps=self.num_mis_steps)

        # Optimizers
        opt_joint, opt_proposal = self.optimizers()

        # Update joint model
        opt_joint.zero_grad()
        loss_joint = self.joint_model(x, h)
        self.manual_backward(loss_joint)
        opt_joint.step()

        # Update proposal model
        opt_proposal.zero_grad()
        loss_proposal = self.proposal_model(h, x)
        self.manual_backward(loss_proposal)
        opt_proposal.step()

        self.log("train/loss_joint", loss_joint, prog_bar=True)
        self.log("train/loss_proposal", loss_proposal, prog_bar=True)

    # ========================= Validation =========================

    def on_validation_start(self):
        # For validation
        if self.log_codebook_utilization_valid:
            self.num_latent_vars = self.proposal_model.num_latent_vars
            self.num_categories = self.proposal_model._num_categories
            self.codebook_size = math.prod(self.num_categories)
            self.codebook_counter = torch.zeros(
                self.codebook_size, dtype=torch.int32, device=self.device
            )

    def validation_step(self, batch, batch_idx):
        x, _, idx = batch  # x: [B, D], idx: [B,]

        nll = -self.get_nll(x, idx=idx)

        self.log("valid/nll", nll.mean(), prog_bar=True, sync_dist=True)

        if batch_idx == 0:
            self.validation_step_outputs.append(x[:16])

        if self.log_codebook_utilization_valid:
            # Update codebook counter
            h = self.proposal_model.sample_latent(x, num_samples=1, encoded=False)
            # Calculate 1D indices from multi-dimensional categorical latent variables
            weights = torch.tensor(
                [
                    math.prod(self.num_categories[i + 1 :])
                    for i in range(self.num_latent_vars)
                ],
                device=self.device,
                dtype=torch.float32,
            )
            # Calculate indices
            indices = torch.matmul(h.float(), weights)  # [B]
            indices = indices.long().squeeze(-1)  # [B]
            self.codebook_counter.index_add_(
                0, indices, torch.ones_like(indices, dtype=torch.int32)
            )

        return {"valid_img": x[:16]}

    def on_validation_epoch_end(self):
        # Show some reconstruction results
        x = self.validation_step_outputs[0]  # [16, D]
        x_hat = self.forward(x)  # [16, D]

        # show original images
        grid_orig = torchvision.utils.make_grid(x.view(-1, 1, 28, 28), nrow=4)
        self.logger.experiment.add_image(
            "valid/original_images", grid_orig, self.current_epoch
        )
        # show reconstructed images
        grid_recon = torchvision.utils.make_grid(x_hat.view(-1, 1, 28, 28), nrow=4)
        self.logger.experiment.add_image(
            "valid/reconstructed_images", grid_recon, self.current_epoch
        )

        self.validation_step_outputs.clear()  # free memory

        if self.log_codebook_utilization_valid:
            # Compute and log codebook utilization
            if dist.is_available() and dist.is_initialized():
                # gather codebook_counter from all ranks
                dist.all_reduce(self.codebook_counter, op=dist.ReduceOp.SUM)

            used_codewords = torch.sum(self.codebook_counter > 0).item()
            utilization_rate = used_codewords / self.codebook_size * 100
            self.log(
                "valid/codebook_utilization",
                utilization_rate,
                prog_bar=True,
                sync_dist=True,  # ensure correct logging in distributed setting
            )

            # Plot codebook usage distribution
            codebook_counter_cpu = self.codebook_counter.cpu().numpy()
            codebook_counter_cpu = np.sort(codebook_counter_cpu)[::-1]
            plt.figure(figsize=(12, 6))
            plt.bar(range(self.codebook_size), codebook_counter_cpu, width=1.0)
            plt.xlabel("Codeword Index")
            plt.ylabel("Usage Count")
            plt.text(
                self.codebook_size * 0.7,
                max(codebook_counter_cpu) * 0.9,
                f"Used codewords: {used_codewords}/{self.codebook_size}\nUtilization rate: {utilization_rate:.4f}%",
                fontsize=12,
                bbox=dict(facecolor="white", alpha=0.5),
            )

            plt.title("Codebook Usage Distribution")
            plt.tight_layout()
            self.logger.experiment.add_figure(
                "valid/codebook_usage_distribution",
                plt.gcf(),
                self.current_epoch,
            )
            plt.close()

            # Reset codebook counter for next epoch
            self.codebook_counter.zero_()

    def get_nll(self, x, idx):

        # MISampling step
        h = self.sampler.sample(x, idx=idx, num_steps=self.num_mis_steps)

        # log p(x) ~ log p(x,h) - log q(h|x)
        log_nll = self.joint_model.log_joint_prob(
            x, h
        ) - self.proposal_model.log_conditional_prob(h, x)

        return log_nll.detach().cpu().numpy()

    # ========================= Testing =========================

    def on_test_start(self):
        # For testing

        self.num_latent_vars = self.proposal_model.num_latent_vars
        self.num_categories = self.proposal_model._num_categories
        self.codebook_size = math.prod(self.num_categories)
        self.codebook_counter = torch.zeros(
            self.codebook_size, dtype=torch.int32, device=self.device
        )
        self.codebook_multi_dim_indices = torch.zeros(
            (self.codebook_size, self.num_latent_vars),
            dtype=torch.int32,
            device=self.device,
        )

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        x, _, idx = batch  # x: [B, D], idx: [B,]

        nll = -self.get_nll(x, idx=idx)

        self.log("test/nll", nll.mean(), prog_bar=True, sync_dist=True)

        # Update codebook counter
        h = self.proposal_model.encode(x)
        # Calculate 1D indices from multi-dimensional categorical latent variables

        # Calculate indices
        indices = encode_multidim_to_index(h, self.num_categories)  # [B]
        
        self.codebook_counter.index_add_(
            0, indices, torch.ones_like(indices, dtype=torch.int32)
        )
        self.codebook_multi_dim_indices[indices] = h.long()

    def on_test_epoch_end(self):
        if dist.is_available() and dist.is_initialized():
            # gather codebook_counter from all ranks
            dist.all_reduce(self.codebook_counter, op=dist.ReduceOp.SUM)

        used_codewords = torch.sum(self.codebook_counter > 0).item()
        utilization_rate = used_codewords / self.codebook_size * 100
        self.log(
            "test/codebook_utilization",
            utilization_rate,
            prog_bar=True,
            sync_dist=True,  # ensure correct logging in distributed setting
        )
        self.log("test/used_codewords", used_codewords, prog_bar=True, sync_dist=True)

        # Plot codebook usage distribution
        fig_dict = plot_codebook_usage_distribution(
            codebook_counter=self.codebook_counter.cpu().numpy(),
            codebook_size=self.codebook_size,
            used_codewords=used_codewords,
            utilization_rate=utilization_rate,
            tag_prefix="test",
            save_to_disk=False,
        )
        for tag, fig in fig_dict.items():
            self.logger.experiment.add_figure(
                tag,
                fig,
                self.current_epoch,
            )

        # Reset codebook counter for next epoch
        self.codebook_counter.zero_()
        self.codebook_multi_dim_indices.zero_()

    # ========================= Checkpointing =========================

    def on_save_checkpoint(self, checkpoint):
        # Ensure cache is synced before saving
        if getattr(self.sampler, "use_cache", False):
            # sync across ranks so rank0 saves consistent cache
            self.sampler.sync_cache()

            # Only rank 0 needs to put sampler state into checkpoint, but returning from hook
            # is executed on all ranks. We place sampler state into checkpoint dict.
            sampler_state = self.sampler.state_dict()
            # store under a known key
            checkpoint["sampler_state"] = sampler_state

    def on_load_checkpoint(self, checkpoint: dict):
        # Load sampler cache from checkpoint if present
        if getattr(self.sampler, "use_cache", False):
            self.sampler.load_state_dict(checkpoint["sampler_state"])
            # after load, ensure sampler cache is on correct device
            self.sampler.to(self.device)
            # broadcast loaded cache to all ranks to be safe
            if dist.is_available() and dist.is_initialized():
                # we expect that checkpoint was saved from rank0 and all ranks loaded same dict,
                # but ensure everyone has the same in distributed environment
                self.sampler.sync_cache()

    @classmethod
    def load_model(cls, config_path: str, checkpoint_path: str, device: str = "cpu"):
        """Load model from config and checkpoint paths"""
        from omegaconf import OmegaConf

        config = OmegaConf.load(config_path)
        model = cls.load_from_checkpoint(checkpoint_path, **config.model.init_args)
        model.to(device)
        model.sampler.to(device)
        return model.eval()
