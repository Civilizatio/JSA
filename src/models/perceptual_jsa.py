# src/models/perceptual_jsa.py
"""Three-stage JSA with perceptual / energy-based reconstruction losses.

The implementation follows the task description in ``docs/task.md`` closely:

* **Stage 1** trains the decoder-side distortion model and the proposal model using only the
  positive phase.
* **Stage 2** freezes stage-1 sampling behaviour and pretrains the autoregressive prior on the
  sampled latent codes.
* **Stage 3** runs full energy-based learning with a negative phase driven by Langevin dynamics
  over ``x`` and MIS / optional NCG updates over ``h``.

The stages can either be forced explicitly (recommended for the intended workflow) or resolved
from epoch boundaries for a single-run schedule.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch
import dataclasses
import torch.distributed as dist


from src.base.base_dataset import JsaDataset
from src.base.base_sampler import BaseSampler
from src.models.jsa import JSA
from src.modules.jsa.joint_model import EnergyOutput, JointLossOutput
from src.modules.jsa.prior_model import PriorAnalysisOutput
from src.utils.file_logger import get_file_logger
from src.utils.instantiate_utils import instantiate
from src.utils.perceptual_stage import PerceptualTrainingStage


class PerceptualJSA(JSA):
    """Three-stage Perceptual JSA training module. See class docstring for details.
        The implementation is designed to be flexible and extensible, allowing for custom samplers, logging, and training schedules. It also includes careful handling of optimizers for different model components and supports checkpointing of the joint negative sampler state.

    Arguments
    ---------
    joint_model: nn.Module
        The joint energy model to be trained. Must implement the necessary interfaces for computing energies and losses.
    proposal_model: Optional[nn.Module]
        An optional proposal model for sampling latent codes. If None, no proposal model will be used and the sampler will rely solely on the joint model's distortion for proposals.
    sampler: Any
        The sampler used for generating positive samples of ``h`` given ``x``. This is typically a MISampler instance.
    langevin_sampler: Optional[Any]
        An optional sampler for generating negative samples of ``x`` given ``h``. This is typically a LangevinSampler instance. If None, a default Langevin sampler will be used in stage 3.
    ncg_sampler: Optional[Any]
        An optional sampler for generating negative samples of ``h`` given ``x``. This is typically an NCGSampler instance. If None, no NCG updates will be performed in stage 3.
    joint_negative_sampler: Optional[Any]
        An optional sampler that jointly samples negative (x, h) pairs. This can be used to implement more complex negative sampling strategies that coordinate updates to both variables. If None, a default alternating Langevin + MISampling strategy will be used in stage 3.
    base_lr_joint: float
        Base learning rate for the joint model components (distortion and prior). This will be scaled by the effective batch size.
    base_lr_prior: float
        Base learning rate for the prior model component. This will be scaled by the effective batch size.
    base_lr_proposal: float
        Base learning rate for the proposal model. This will be scaled by the effective batch size.
    num_mis_steps: int
        Number of MIS steps to run for each positive sample generation.
    cache_start_epoch: int
        Epoch at which to start caching positive samples for replay. If <= 0, caching is disabled.
    init_from_ckpt: Optional[str]
        Path to a checkpoint to initialize the model from. This can be used to resume training or to initialize from a pretrained model.
    init_mode: str
        Mode for loading the checkpoint. One of {"resume", "warm_start"}. "resume" will load the entire training state including optimizers and samplers. "scratch" will only load the model weights and ignore optimizer and sampler states. "pretrain" will load model weights but not optimizer states, allowing for fine-tuning from a pretrained model.
    init_strict: bool
        Whether to strictly enforce that the checkpoint state dict keys match the model's state dict keys when loading. If False, missing or unexpected keys will be ignored with a warning.
    global_only_steps: int
        Number of initial steps to train with only the global energy (no distortion/prior decomposition). This can help stabilize training in the early stages.
    block_strategy_prob: float
        Probability of using a "block" strategy for MIS sampling, where entire blocks of latent variables are proposed together. This can be used to encourage more global proposals.
    force_stage: Optional[Union[PerceptualTrainingStage, int, str]]
        If specified, forces the training to run in a specific stage regardless of epoch. Can be an integer (1, 2, or 3), a string alias ("decoder", "prior", "full"), or a PerceptualTrainingStage enum value.
    auto_stage_epochs: Optional[Sequence[int]]
        If force_stage is not set, this tuple of two integers specifies the number of epochs to run in stage 1 and stage 2 before automatically transitioning to the next stage. For example, (5, 10) would run stage 1 for 5 epochs, then stage 2 for 10 epochs, and then stay in stage 3 for the rest of training.
    stage3_negative_rounds: int
        Number of rounds of negative sampling to perform in stage 3 for each positive sample. Each round consists of one update to x (e.g., Langevin step) and one update to h (e.g., MIS step and optional NCG step).
    parallel_mis_sampling: bool
        Whether to run MIS sampling in parallel for all samples in the batch (if True) or sequentially (if False). Parallel sampling can be more efficient but may use more memory.
    train_proposal_in_stage1: bool
        Whether to train the proposal model in stage 1 along with the distortion model. If False, the proposal model will not receive gradient updates in stage 1.
    train_proposal_in_stage3: bool
        Whether to train the proposal model in stage 3 along with the distortion and prior models. If False, the proposal model will not receive gradient updates in stage 3.
    num_validation_mis_steps: Optional[int]
        If specified, overrides the number of MIS steps to run during validation. This can be used to reduce validation time by using fewer MIS steps than in training.
    """

    def __init__(
        self,
        joint_model,
        proposal_model,
        sampler,
        langevin_sampler: Any = None,
        ncg_sampler: Any = None,
        joint_negative_sampler: Any = None,
        base_lr_joint: float = 1e-4,
        base_lr_prior: float = 1e-4,
        base_lr_proposal: float = 1e-4,
        weight_decay_prior: float = 0.01,
        num_mis_steps: int = 3,
        cache_start_epoch: int = 0,
        init_from_ckpt: str = None,
        init_mode: str = "resume",
        init_strict: bool = False,
        global_only_steps: int = 10000,
        block_strategy_prob: float = 0.5,
        force_stage=None,
        auto_stage_epochs: Optional[Sequence[int]] = None,
        stage3_negative_rounds: int = 1,
        parallel_mis_sampling: bool = True,
        train_proposal_in_stage1: bool = True,
        train_proposal_in_stage3: bool = True,
        num_validation_mis_steps: Optional[int] = None,
    ):
        super().__init__(
            joint_model=joint_model,
            proposal_model=proposal_model,
            sampler=sampler,
            gan_loss=None,
            base_lr_joint=base_lr_joint,
            base_lr_proposal=base_lr_proposal,
            base_lr_discriminator=0.0,
            num_mis_steps=num_mis_steps,
            cache_start_epoch=cache_start_epoch,
            init_from_ckpt=init_from_ckpt,
            init_mode=init_mode,
            init_strict=init_strict,
            sigma_scheduler=None,
            global_only_steps=global_only_steps,
            block_strategy_prob=block_strategy_prob,
        )
        self.base_lr_prior = base_lr_prior
        self.lr_prior = base_lr_prior
        self.weight_decay_prior = weight_decay_prior

        self.force_stage = PerceptualTrainingStage.from_value(force_stage)
        if auto_stage_epochs is None:
            auto_stage_epochs = [0, 0]
        if len(auto_stage_epochs) != 2:
            raise ValueError(
                "auto_stage_epochs must contain exactly two integers: [stage1_epochs, stage2_epochs]."
            )
        self.auto_stage_epochs = tuple(int(v) for v in auto_stage_epochs)
        self.stage3_negative_rounds = int(stage3_negative_rounds)
        self.parallel_mis_sampling = bool(parallel_mis_sampling)
        self.train_proposal_in_stage1 = bool(train_proposal_in_stage1)
        self.train_proposal_in_stage3 = bool(train_proposal_in_stage3)
        self.num_validation_mis_steps = num_validation_mis_steps

        self.langevin_sampler: Optional[BaseSampler] = (
            instantiate(langevin_sampler, joint_model=self.joint_model)
            if langevin_sampler is not None
            else None
        )
        self.ncg_sampler: Optional[BaseSampler] = (
            instantiate(
                ncg_sampler,
                joint_model=self.joint_model,
                proposal_model=self.proposal_model,
            )
            if ncg_sampler is not None
            else None
        )
        self.joint_negative_sampler: Optional[BaseSampler] = (
            instantiate(
                joint_negative_sampler,
                joint_model=self.joint_model,
                proposal_model=self.proposal_model,
                h_sampler=self.sampler,
                x_sampler=self.langevin_sampler,
                ncg_sampler=self.ncg_sampler,
            )
            if joint_negative_sampler is not None
            else None
        )

        self.grad_norm_modules = {
            "distortion_model": getattr(
                self.joint_model, "distortion_model", self.joint_model
            ),
            "prior_model": getattr(self.joint_model, "prior_model", self.joint_model),
            "proposal_model": self.proposal_model,
        }
        # optimizer_slots keeps track of which optimizer index corresponds to which model component, since not all components may be present and the optimizers list is constructed dynamically in configure_optimizers
        #! This is a very important detail: you should not change it to a dictionary that directly stores the optimizers,
        #! because the `glboal_step` is updated when we call `self.optimizers()` and we want to make sure that the correct optimizer is stepped for each component in each training stage, even as the optimizers are created dynamically based on which model components are present.
        self._optimizer_slots = {
            "distortion": None,
            "prior": None,
            "proposal": None,
        }

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------
    def _resolve_stage(self, epoch: Optional[int] = None):
        if self.force_stage is not None:
            return self.force_stage
        epoch = self.current_epoch if epoch is None else int(epoch)
        stage1_epochs, stage2_epochs = self.auto_stage_epochs
        if epoch < stage1_epochs:
            return PerceptualTrainingStage.DECODER_PRETRAIN
        if epoch < stage1_epochs + stage2_epochs:
            return PerceptualTrainingStage.PRIOR_PRETRAIN
        return PerceptualTrainingStage.FULL_EBM

    def _apply_runtime_stage(self):
        stage = self._resolve_stage()
        if hasattr(self.joint_model, "set_stage"):
            self.joint_model.set_stage(stage)
        return stage

    @staticmethod
    def _strip_sample_dim(h):
        if (
            h.dim() >= 5 and h.shape[1] == 1
        ):  # Assuming shape [B, 1, H, W, num_latent_vars], treat as token indices and remove the singleton channel dimension
            return h[:, 0]  # [B, H, W, num_latent_vars]
        return h

    def _validation_mis_steps(self):
        if self.num_validation_mis_steps is not None:
            return int(self.num_validation_mis_steps)
        return int(self.num_mis_steps)

    def setup(self, stage=None):
        super().setup(stage=stage)
        device = self.device
        if self.langevin_sampler is not None and hasattr(self.langevin_sampler, "to"):
            self.langevin_sampler.to(device)
        if self.ncg_sampler is not None and hasattr(self.ncg_sampler, "to"):
            self.ncg_sampler.to(device)
        if self.joint_negative_sampler is not None and hasattr(
            self.joint_negative_sampler, "to"
        ):
            self.joint_negative_sampler.to(device)

    # ------------------------------------------------------------------
    # Logging / optimizer helpers
    # ------------------------------------------------------------------

    def on_train_start(self):
        super().on_train_start()
        if self.train_logger is not None:
            self.train_logger.info(
                f"PerceptualJSA stage mode: force_stage={self.force_stage}, auto_stage_epochs={self.auto_stage_epochs}"
            )
            if hasattr(self.joint_model, "distortion_model"):
                self.train_logger.info("Distortion Model Structure:")
                self.train_logger.info(self.joint_model.distortion_model)
            if hasattr(self.joint_model, "prior_model"):
                self.train_logger.info("Prior Model Structure:")
                self.train_logger.info(self.joint_model.prior_model)

    def configure_optimizers(self):
        batch_size = (
            self.trainer.datamodule.batch_size
            if self.trainer and self.trainer.datamodule
            else 1
        )
        num_devices = (
            self.trainer.num_devices if self.trainer and self.trainer.num_devices else 1
        )
        accumulated_batches = (
            self.trainer.accumulate_grad_batches if self.trainer else 1
        )
        effective_batch_size = batch_size * num_devices * accumulated_batches
        self.lr_joint = self.base_lr_joint * effective_batch_size
        self.lr_prior = self.base_lr_prior * effective_batch_size
        self.lr_proposal = self.base_lr_proposal * effective_batch_size

        if self.train_logger is not None:
            self.train_logger.info(
                "Configuring PerceptualJSA optimizers with effective batch size %s, "
                "lr_joint=%s, lr_prior=%s, lr_proposal=%s",
                effective_batch_size,
                self.lr_joint,
                self.lr_prior,
                self.lr_proposal,
            )

        optimizers = []

        distortion_params = [
            p for p in self.joint_model.distortion_model.parameters() if p.requires_grad
        ]
        if len(distortion_params) > 0:
            self._optimizer_slots["distortion"] = len(optimizers)
            optimizers.append(torch.optim.Adam(distortion_params, lr=self.lr_joint))

        # Prior parameter grouping for AdamW
        if hasattr(self.joint_model.prior_model, "parameters"):
            prior_params = list(self.joint_model.prior_model.parameters())
            if len([p for p in prior_params if p.requires_grad]) > 0:
                self._optimizer_slots["prior"] = len(optimizers)
                
                decay_params = []
                no_decay_params = []
                
                # Modules that should have weight decay applied to their parameters (e.g., Linear layers)
                import torch.nn as nn
                whitelist_weight_modules = (nn.Linear,)
                # Modules that should not have weight decay applied to their parameters (e.g., LayerNorm, Embedding)
                blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)
                
                name_to_module = {n: m for n, m in self.joint_model.prior_model.named_modules()}
                
                for name, param in self.joint_model.prior_model.named_parameters():
                    if not param.requires_grad:
                        continue
                        
                    if name.endswith("pos_emb"):
                        no_decay_params.append(param)
                        continue
                        
                    if name.endswith("bias"):
                        no_decay_params.append(param)
                        continue
                        
                    parent_name = ".".join(name.split(".")[:-1]) if "." in name else ""
                    parent_module = name_to_module.get(parent_name, None)
                    
                    if isinstance(parent_module, blacklist_weight_modules):
                        no_decay_params.append(param)
                    elif isinstance(parent_module, whitelist_weight_modules) and name.endswith("weight"):
                        decay_params.append(param)
                    else:
                        no_decay_params.append(param)
                        
                optim_groups = [
                    {"params": decay_params, "weight_decay": self.weight_decay_prior},
                    {"params": no_decay_params, "weight_decay": 0.0},
                ]
                
                optimizers.append(torch.optim.AdamW(optim_groups, lr=self.lr_prior, betas=(0.9, 0.95)))

        proposal_params = [
            p for p in self.proposal_model.parameters() if p.requires_grad
        ]
        if len(proposal_params) > 0:
            self._optimizer_slots["proposal"] = len(optimizers)
            optimizers.append(torch.optim.Adam(proposal_params, lr=self.lr_proposal))

        return optimizers

    def _optimizer_dict(self) -> Dict[str, torch.optim.Optimizer]:
        optimizers = self.optimizers()
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]
        result = {}
        for name, idx in self._optimizer_slots.items():
            if idx is not None:
                result[name] = optimizers[idx]
        return result

    def _log_component_summary(
        self,
        prefix: str,
        summary: Dict[str, torch.Tensor] | dataclasses.dataclass | Any,
    ):
        if dataclasses.is_dataclass(summary):
            summary = dataclasses.asdict(summary)
        for key, value in summary.items():
            if not torch.is_tensor(value):
                value = torch.tensor(float(value), device=self.device)
            # IMPORTANT: always detach tensors before logging to avoid memory leaks
            self.log(
                f"{prefix}/{key}",
                value.detach(),
                prog_bar=False,
                logger=True,
                sync_dist=False,
            )

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------
    def _sample_positive_h(self, x, idx, stage):
        """Samples positive latent codes h for a given input x and training stage.
        The sampling strategy can depend on the stage, allowing for different proposal mechanisms in different stages of training.
        """
        if hasattr(self.joint_model, "set_stage"):
            self.joint_model.set_stage(stage)
        return self.sampler.sample(
            x,
            idx=idx,
            num_steps=self.num_mis_steps,
            parallel=self.parallel_mis_sampling,
            return_all=False,
            strategy=self._choose_sampling_strategy(),
        )

    def _default_negative_sample(self, x_real, h_pos):
        """Default negative sampling strategy for stage 3 when no joint_negative_sampler is provided.
        This consists of alternating updates to x and h, where x is updated with Langevin dynamics and h is updated with the MIS sampler and optional NCG sampler.
        The number of rounds of updates is controlled by `stage3_negative_rounds`.

        NOTE:x is initialized by adding small Gaussian noise to the real x, and h is initialized from the positive sample.
        """
        if self.langevin_sampler is None:
            raise RuntimeError(
                "Stage 3 requires a `langevin_sampler` or a `joint_negative_sampler`."
            )
        x_neg = x_real + 0.01 * torch.randn_like(x_real)
        x_neg = x_neg.clamp(-1.0, 1.0)
        h_neg = self._strip_sample_dim(h_pos).detach().clone()
        for _ in range(self.stage3_negative_rounds):
            x_neg = self.langevin_sampler.sample(x_neg, h_neg)
            h_neg = self.sampler.sample(
                x_neg,
                idx=None,
                num_steps=1,
                parallel=False,
                return_all=False,
                strategy="none",
            )
            h_neg = self._strip_sample_dim(h_neg)
            if self.ncg_sampler is not None:
                h_neg = self.ncg_sampler.sample(x_neg, h_neg)
        return x_neg.detach(), h_neg.detach()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):

        stage = self._apply_runtime_stage()
        x = self.get_input(batch, JsaDataset.IMAGE_KEY)
        idx = self.get_input(batch, JsaDataset.INDEX_KEY)
        optimizers = self._optimizer_dict()
        zero = torch.tensor(0.0, device=self.device)

        h_pos = self._sample_positive_h(x, idx=idx, stage=stage)
        self.log(
            "train/stage", torch.tensor(float(stage), device=self.device), prog_bar=True
        )

        # Stage 1: pretrain distortion model (and optionally proposal model) with positive samples only.
        if stage == PerceptualTrainingStage.DECODER_PRETRAIN:
            opt_dist = optimizers["distortion"]
            opt_dist.zero_grad()
            loss_joint_out: JointLossOutput = self.joint_model.get_loss(
                x,
                h_pos,
                stage=stage,
                return_forward=True,
                return_components=True,
            )
            loss_joint = loss_joint_out.loss
            summary = loss_joint_out.components
            self.manual_backward(loss_joint)
            opt_dist.step()

            if self.train_proposal_in_stage1 and "proposal" in optimizers:
                opt_prop = optimizers["proposal"]
                opt_prop.zero_grad()
                loss_prop = self.proposal_model.get_loss(h_pos, x)
                self.manual_backward(loss_prop)
                opt_prop.step()
            else:
                loss_prop = zero

            self.log("train/loss_joint", loss_joint, prog_bar=True)
            self.log("train/loss_proposal", loss_prop, prog_bar=True)
            self._log_component_summary("train", summary)
            return {
                "loss_joint": loss_joint.detach(),
                "loss_proposal": loss_prop.detach(),
            }

        # Stage 2: pretrain prior model with positive samples only.
        if stage == PerceptualTrainingStage.PRIOR_PRETRAIN:
            opt_prior = optimizers["prior"]
            opt_prior.zero_grad()
            h_for_prior = self._strip_sample_dim(h_pos)
            loss_prior = self.joint_model.prior_model.get_loss(h_for_prior)
            self.manual_backward(loss_prior)
            opt_prior.step()

            self.log("train/loss_prior", loss_prior, prog_bar=True)
            if hasattr(self.joint_model.prior_model, "analyze"):
                prior_stats = self.joint_model.prior_model.analyze(h_for_prior)
                self._log_component_summary("train", prior_stats)
            return {"loss_prior": loss_prior.detach()}

        # Stage 3: full energy-based training with negative phase.
        if stage == PerceptualTrainingStage.FULL_EBM:
            if self.joint_negative_sampler is not None:
                x_neg, h_neg = self.joint_negative_sampler.sample(x, h_pos=h_pos)
            else:
                x_neg, h_neg = self._default_negative_sample(x, h_pos)

            opt_dist = optimizers.get("distortion")
            opt_prior = optimizers.get("prior")
            if opt_dist is not None:
                opt_dist.zero_grad()
            if opt_prior is not None:
                opt_prior.zero_grad()

            out: EnergyOutput = self.joint_model.energy_multiple_samples(
                x, h_pos, stage=stage
            )
            pos_energy_all, pos_components = out.energy, out.components
            pos_energy = pos_energy_all.mean()
            out: EnergyOutput = self.joint_model.energy(x_neg, h_neg, stage=stage)
            neg_energy, neg_components = out.energy, out.components
            neg_energy = neg_energy.mean()

            scale_factor = (
                x.numel() / x.shape[0]
            )  # Scale the loss by the number of elements per sample to keep it invariant to batch size
            loss_joint_unscaled = pos_energy - neg_energy
            loss_joint = loss_joint_unscaled / scale_factor
            self.manual_backward(loss_joint)
            if opt_dist is not None:
                opt_dist.step()
            if opt_prior is not None:
                opt_prior.step()

            if self.train_proposal_in_stage3 and "proposal" in optimizers:
                opt_prop = optimizers["proposal"]
                opt_prop.zero_grad()
                loss_prop = self.proposal_model.get_loss(h_pos, x)
                self.manual_backward(loss_prop)
                opt_prop.step()
            else:
                loss_prop = zero

            self.log("train/loss_joint", loss_joint, prog_bar=True)
            self.log("train/loss_positive_energy", pos_energy, prog_bar=True)
            self.log("train/loss_negative_energy", neg_energy, prog_bar=True)
            self.log("train/loss_proposal", loss_prop, prog_bar=True)
            self._log_component_summary(
                "train/positive",
                {key: value.mean() for key, value in pos_components.items()},
            )
            self._log_component_summary(
                "train/negative",
                {key: value.mean() for key, value in neg_components.items()},
            )
            return {
                "loss_joint": loss_joint.detach(),
                "loss_proposal": loss_prop.detach(),
                "positive_energy": pos_energy.detach(),
                "negative_energy": neg_energy.detach(),
            }

        #! If the stage is somehow not recognized (which shouldn't happen), return an empty dict.
        return {}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        acceptance_rate = self.sampler.get_acceptance_rate()
        self.log("train/mis_acceptance_rate", acceptance_rate, prog_bar=False)
        self.log(
            "train/current_stage",
            torch.tensor(float(self._resolve_stage()), device=self.device),
            prog_bar=False,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        stage = self._apply_runtime_stage()
        x = self.get_input(batch, JsaDataset.IMAGE_KEY)
        idx = self.get_input(batch, JsaDataset.INDEX_KEY)

        h = self.sampler.sample(
            x,
            idx=idx,
            num_steps=self._validation_mis_steps(),
            parallel=False,
            return_all=False,
            strategy="none",
        )
        h_last = self._strip_sample_dim(h)
        x_rec = self.decode(h_last)

        recon_l1 = torch.nn.functional.l1_loss(x_rec, x, reduction="mean")
        recon_mse = torch.nn.functional.mse_loss(x_rec, x, reduction="mean")

        output: EnergyOutput = self.joint_model.energy_multiple_samples(
            x, h, stage=stage
        )
        pos_energy = output.energy.mean()
        components = output.components

        self.log("valid/positive_energy", pos_energy, prog_bar=True, sync_dist=True)
        self.log("valid/recon_l1", recon_l1, prog_bar=True, sync_dist=True)
        self.log("valid/recon_mse", recon_mse, prog_bar=True, sync_dist=True)
        for key, value in components.items():
            self.log(f"valid/{key}", value.mean(), prog_bar=False, sync_dist=True)

        if hasattr(self.joint_model.prior_model, "analyze"):
            prior_stats: PriorAnalysisOutput = self.joint_model.prior_model.analyze(
                h_last
            )

            for key, value in dataclasses.asdict(prior_stats).items():
                self.log(f"valid/{key}", value, prog_bar=False, sync_dist=True)

    # ------------------------------------------------------------------
    # Checkpoint extras
    # ------------------------------------------------------------------
    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)

        if self.joint_negative_sampler is not None:
            checkpoint["joint_negative_sampler_state"] = (
                self.joint_negative_sampler.state_dict()
            )

    def on_load_checkpoint(self, checkpoint: dict):
        super().on_load_checkpoint(checkpoint)
        if (
            self.joint_negative_sampler is not None
            and "joint_negative_sampler_state" in checkpoint
        ):
            self.joint_negative_sampler.load_state_dict(
                checkpoint["joint_negative_sampler_state"]
            )

    # ------------------------------------------------------------------
    # Callback Utilities
    # ------------------------------------------------------------------
    @torch.no_grad()
    def safe_decode(self, h):
        """Decodes latent codes h into images, safely handling any out-of-bounds tokens
        (like the sos_token from the prior model) by clamping them to the valid range.
        """
        if hasattr(self.proposal_model, "num_categories"):
            num_categories = self.proposal_model.num_categories
            if isinstance(num_categories, int):
                h = torch.clamp(h, min=0, max=num_categories - 1)
            elif isinstance(num_categories, (list, tuple)) and h.shape[-1] == len(
                num_categories
            ):
                h_clamped = h.clone()
                for i, max_c in enumerate(num_categories):
                    h_clamped[..., i] = torch.clamp(h[..., i], min=0, max=max_c - 1)
                h = h_clamped
        return self.decode(h)

    @torch.no_grad()
    def log_images(
        self,
        batch,
        temperature=1.0,
        top_k=100,
        top_p=1.0,
        callback=lambda k: None,
        **kwargs,
    ):
        """Implement the logic to generate and log images during validation.
        In Stage 1, it logs inputs and reconstructions.
        In Stage 2 and Stage 3, it adds generated samples from the prior model.
        """
        log = super().log_images(batch, **kwargs)

        stage = self._apply_runtime_stage()

        # Only in Stage 2 (PRIOR_PRETRAIN) and Stage 3 (FULL_EBM) do we consider prior samples
        if stage in [
            PerceptualTrainingStage.PRIOR_PRETRAIN,
            PerceptualTrainingStage.FULL_EBM,
        ]:
            prior_model = getattr(self.joint_model, "prior_model", None)

            if prior_model is not None and hasattr(prior_model, "sample"):
                x = log["inputs"]

                # To get the spatial shape, we encode the current image using proposal model
                h_encoded = self.proposal_model.encode(x)
                spatial_shape = h_encoded.shape[1:-1]

                # 1. Generate samples from scratch
                h_sampled = prior_model.sample(
                    batch_size=x.shape[0],
                    spatial_shape=spatial_shape,
                    device=x.device,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                x_sampled = self.safe_decode(h_sampled)
                log["samples_from_scratch"] = x_sampled

                # 2. Generate deterministic samples (greedy decoding: temperature=0)
                h_deterministic = prior_model.sample(
                    batch_size=x.shape[0],
                    spatial_shape=spatial_shape,
                    device=x.device,
                    temperature=0.0,
                    top_k=None,
                    top_p=1.0,
                )
                x_deterministic = self.safe_decode(h_deterministic)
                log["samples_deterministic"] = x_deterministic

        return log
