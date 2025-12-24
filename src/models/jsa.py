# src/models/jsa.py
import torch
from lightning.pytorch import LightningModule
from hydra.utils import instantiate
import torchvision
import torch.distributed as dist

from src.samplers.misampler import MISampler
from src.base.base_jsa_modules import BaseJointModel, BaseProposalModel
from src.utils.codebook_utils import (
    encode_multidim_to_index,
    plot_codebook_usage_distribution,
)
from src.utils.controllers import ParameterController

from src.models.components.losses import JSAGANLoss
import math
import numpy as np
import matplotlib.pyplot as plt


class JSA(LightningModule):
    def __init__(
        self,
        joint_model: BaseJointModel,
        proposal_model: BaseProposalModel,
        sampler,
        gan_loss: JSAGANLoss = None,
        lr_joint=1e-3,
        lr_proposal=1e-3,
        lr_discriminator=1e-4,
        num_mis_steps=3,
        cache_start_epoch=0,
        init_from_ckpt: str = None,
        init_mode: str = "resume",  # "resume" or "warm_start"
        init_strict: bool = False,
        adaptive_sigma: bool = False,
        target_acceptance_range: tuple = (0.2, 0.5),
        sigma_controller_rate: float = 0.01,
        sigma_min: float = 0.01,
        sigma_max: float = 1.0,
    ):
        super().__init__()
        # self.save_hyperparameters(ignore=["joint_model", "proposal_model", "sampler", "gan_loss"])

        self.joint_model: BaseJointModel = joint_model
        self.proposal_model: BaseProposalModel = proposal_model
        self.sampler: MISampler = instantiate(
            sampler,
            joint_model=self.joint_model,
            proposal_model=self.proposal_model,
        )

        self.gan_loss: JSAGANLoss = gan_loss
        self.automatic_optimization = False
        self.lr_joint = lr_joint
        self.lr_proposal = lr_proposal
        self.lr_discriminator = lr_discriminator
        self.num_mis_steps = num_mis_steps
        self.cache_start_epoch = cache_start_epoch

        self.init_from_ckpt = init_from_ckpt
        self.init_mode = init_mode
        self.init_strict = init_strict
        self._weights_loaded = False  # guard to ensure weights are loaded only once

        self.adaptive_sigma = adaptive_sigma
        if self.adaptive_sigma:
            self.sigma_controller = ParameterController(
                param_name="mis_sigma",
                target_range=target_acceptance_range,
                adjustment_rate=sigma_controller_rate,
                min_val=sigma_min,
                max_val=sigma_max,
                mode="exponential",
                direction="inverse",
            )
        else:
            self.sigma_controller = None

        # For visualization during validation
        self.validation_step_outputs = []

        self.log_codebook_utilization_valid = True
        self.log_codebook_utilization_test = False

    def setup(self, stage=None):
        device = self.device
        self.sampler.to(device)

    def on_fit_start(self):
        """Warm start or resume from checkpoint if specified

        warm_start: load model weights but not optimizer states

        """
        if self.init_from_ckpt is None or self._weights_loaded:
            return

        if self.init_mode not in ["resume", "warm_start"]:
            raise ValueError(
                f"Unknown init_mode '{self.init_mode}'. Supported modes are 'resume' and 'warm_start'."
            )

        ckpt = torch.load(self.init_from_ckpt, map_location=self.device)

        missing, unexpected = self.load_state_dict(
            ckpt["state_dict"], strict=self.init_strict
        )
        if missing or unexpected:
            print(
                f"While loading weights from '{self.init_from_ckpt}', missing keys: {missing}, unexpected keys: {unexpected}"
            )

        if self.init_mode == "warm_start":
            self.trainer.global_step = 0
            self.trainer.fit_loop.epoch_loop._batches_that_stepped = (
                0  # reset batch count
            )

            # force lr schedulers to be re-initialized
            if self.trainer.lr_schedulers is not None:
                for lr_scheduler in self.trainer.lr_schedulers:
                    lr_scheduler["scheduler"].last_epoch = -1

            # force optimizers to be re-initialized
            optimizers = self.trainer.optimizers
            for optimizer in optimizers:
                optimizer.state = {}

        print(
            f"Model weights loaded from '{self.init_from_ckpt}' with mode '{self.init_mode}'."
        )
        self._weights_loaded = True

    def forward(self, x, idx=None):

        if idx is not None:
            h = self.sampler.sample(
                x, idx=idx, num_steps=self.num_mis_steps
            )  # [B, 1, ..., num_latent_vars]
        else:  # use proposal model directly
            h = self.proposal_model.sample_latent(
                x, num_samples=1
            )  # [B, 1, ..., num_latent_vars]
        h = h.squeeze(1)  # [B, ..., num_latent_vars]
        x_hat = self.joint_model.sample(h=h, num_samples=1).squeeze(1)  # [B, C, H, W]
        return x_hat

    def configure_optimizers(self):
        opt_joint = torch.optim.Adam(self.joint_model.parameters(), lr=self.lr_joint)
        opt_proposal = torch.optim.Adam(
            self.proposal_model.parameters(), lr=self.lr_proposal
        )
        optimizers = [opt_joint, opt_proposal]
        if self.gan_loss is not None:
            opt_discriminator = torch.optim.Adam(
                self.gan_loss.discriminator.parameters(), lr=self.lr_discriminator
            )
            optimizers.append(opt_discriminator)

        return optimizers

    # ========================= Training =========================
    def on_train_start(self):

        # print model structure
        print("Joint Model Structure:")
        print(self.joint_model.net)
        print("Proposal Model Structure:")
        print(self.proposal_model.net)

    def on_train_epoch_start(self):
        if self.current_epoch >= self.cache_start_epoch:
            self.sampler.use_cache = True
        else:
            self.sampler.use_cache = False

        # Reset acceptance stats
        self.sampler.reset_acceptance_stats()

    def training_step(self, batch, batch_idx):
        x, _, idx = batch  # x: [B, C, H, W], idx: [B,]

        # MISampling step
        h = self.sampler.sample(
            x, idx=idx, num_steps=self.num_mis_steps
        )  # [B, 1, ..., num_latent_vars]

        # Optimizers
        if self.gan_loss is not None:
            opt_joint, opt_proposal, opt_discriminator = self.optimizers()
        else:
            opt_joint, opt_proposal = self.optimizers()

        # ============= Update Generator (Joint + Proposal) =============

        # Update joint model
        opt_joint.zero_grad()
        nll_loss, x_hat = self.joint_model.get_loss(
            x, h.squeeze(1), return_forward=True
        )
        total_loss_joint = nll_loss
        if self.gan_loss is not None:

            last_layer = (
                self.joint_model.get_last_layer_weight()
                if hasattr(self.joint_model, "get_last_layer_weight")
                else None
            )

            g_loss, g_log = self.gan_loss(
                inputs=x,
                reconstructions=x_hat,
                optimizer_idx=0,
                global_step=self.global_step,
                last_layer=last_layer,
                nll_loss=nll_loss,
                split="train",
            )
            total_loss_joint = total_loss_joint + g_loss
            self.log_dict(g_log, prog_bar=False)

        self.manual_backward(total_loss_joint)
        opt_joint.step()

        # Update proposal model
        opt_proposal.zero_grad()
        loss_proposal = self.proposal_model.get_loss(h, x)
        self.manual_backward(loss_proposal)
        opt_proposal.step()

        # ============= Update Discriminator =============
        if self.gan_loss is not None:
            opt_discriminator.zero_grad()
            d_loss, d_log = self.gan_loss(
                inputs=x,
                reconstructions=x_hat.detach(),  # detach to avoid gradients to generator
                optimizer_idx=1,
                global_step=self.global_step,
                split="train",
            )
            self.manual_backward(d_loss)
            opt_discriminator.step()
            self.log_dict(d_log, prog_bar=False)

        self.log("train/loss_joint_nll", nll_loss, prog_bar=True)
        self.log("train/loss_proposal", loss_proposal, prog_bar=True)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        accept_rate = self.sampler.get_acceptance_rate()
        self.log("train/mis_acceptance_rate", accept_rate, prog_bar=False)

    def on_train_epoch_end(self):
        if (
            self.adaptive_sigma
            and self.sigma_controller is not None
            and hasattr(self.joint_model, "sigma")
        ):
            current_accept_rate = self.sampler.get_acceptance_rate()
            current_sigma = self.joint_model.sigma.item()

            new_sigma, info = self.sigma_controller.step(
                current_val=current_sigma, metric_val=current_accept_rate
            )

            if info["status"] == "adjusting":
                self.joint_model.sigma.fill_(new_sigma)

            self.log("train/mis_sigma", new_sigma, prog_bar=True)
            self.log("train/mis_sigma_diff", info["diff"], prog_bar=False)

            if info["status"] == "adjusting":
                # print only when adjustment happens
                print(
                    f"[Adaptive Sigma] Rate: {current_accept_rate:.4f} (Target: {self.sigma_controller.min_target}-{self.sigma_controller.max_target}), Sigma: {current_sigma:.4f} -> {new_sigma:.4f}"
                )

    # ========================= Validation =========================

    def on_validation_start(self):
        # For validation
        if self.log_codebook_utilization_valid:
            self.num_latent_vars = self.proposal_model.num_latent_vars
            self.num_categories = self.proposal_model._num_categories
            self.codebook_size = math.prod(self.num_categories)
            self.codebook_counter = torch.zeros(
                self.codebook_size, dtype=torch.long, device=self.device
            )

    def validation_step(self, batch, batch_idx):
        x, _, idx = batch  # x: [B, D], idx: [B,]

        nll = -self.get_nll(x, idx=idx)

        self.log("valid/nll", nll.mean(), prog_bar=True, sync_dist=True)

        if batch_idx == 0:
            self.validation_step_outputs.append(x[:25])

        if self.log_codebook_utilization_valid:
            # Update codebook counter
            h = self.proposal_model.encode(x)
            h = h.view(-1, self.num_latent_vars)  # [B*H*W, num_latent_vars]
            # Calculate 1D indices from multi-dimensional categorical latent variables
            indices = encode_multidim_to_index(h, self.num_categories)  # [B]
            self.codebook_counter.index_add_(
                0, indices, torch.ones_like(indices, dtype=torch.long)
            )

        return {"valid_img": x[:25]}

    def on_validation_epoch_end(self):
        # Show some reconstruction results
        x = self.validation_step_outputs[0]  # [16, D]
        x_hat = self.forward(x)  # [16, D]

        # show original images
        grid_orig = torchvision.utils.make_grid(x, nrow=5)
        self.logger.experiment.add_image(
            "valid/original_images", grid_orig, self.current_epoch
        )
        # show reconstructed images
        grid_recon = torchvision.utils.make_grid(x_hat, nrow=5)
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
            fig_dict = plot_codebook_usage_distribution(
                codebook_counter=self.codebook_counter.cpu().numpy(),
                codebook_size=self.codebook_size,
                used_codewords=used_codewords,
                utilization_rate=utilization_rate,
                tag_prefix="valid",
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

    def get_nll(self, x, idx):

        # MISampling step
        h = self.sampler.sample(
            x, idx=idx, num_steps=self.num_mis_steps, parallel=True
        )  # [B, 1, ..., num_latent_vars]
        h = h.squeeze(1)  # [B, ..., num_latent_vars]

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
            self.codebook_size, dtype=torch.long, device=self.device
        )
        self.codebook_multi_dim_indices = torch.zeros(
            (self.codebook_size, self.num_latent_vars),
            dtype=torch.long,
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
            0, indices, torch.ones_like(indices, dtype=torch.long)
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

    # @classmethod
    # def load_model(cls, config_path: str, checkpoint_path: str, device: str = "cpu"):
    #     """Load model from config and checkpoint paths"""
    #     from omegaconf import OmegaConf

    #     config = OmegaConf.load(config_path)
    #     model = instantiate(config.model)
    #     ckpt = torch.load(checkpoint_path, map_location=device)
    #     model.load_state_dict(ckpt["state_dict"])
    #     # model = cls.load_from_checkpoint(checkpoint_path, **config.model.init_args)
    #     model.to(device)
    #     model.sampler.to(device)
    #     return model.eval()
