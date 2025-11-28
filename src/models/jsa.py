# src/models/jsa.py
import torch
from lightning.pytorch import LightningModule
from hydra.utils import instantiate
import torchvision
import torch.distributed as dist

from src.samplers.misampler import MISampler
from src.base.base_jsa_modules import BaseJointModel, BaseProposalModel


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

        self.validation_step_outputs = []

    def forward(self, x, idx=None):

        if idx is not None:
            idx = idx.tolist()
            h = self.sampler.sample(x, idx=idx, num_steps=self.num_mis_steps)
        else:  # use proposal model directly
            h = self.proposal_model.sample_latent(x, num_samples=1)
            h = h.squeeze(1)
        x_hat = self.joint_model.sample(h=h)
        return x_hat

    def setup(self, stage=None):
        device = self.device
        self.sampler.to(device)

    def configure_optimizers(self):
        opt_joint = torch.optim.Adam(self.joint_model.parameters(), lr=self.lr_joint)
        opt_proposal = torch.optim.Adam(
            self.proposal_model.parameters(), lr=self.lr_proposal
        )
        return [opt_joint, opt_proposal]

    def training_step(self, batch, batch_idx):
        x, _, idx = batch  # x: [B, D], idx: [B,]
        idx = idx.tolist()

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

    def get_nll(self, x, idx):

        # MISampling step
        h = self.sampler.sample(x, idx=idx, num_steps=self.num_mis_steps)

        # log p(x) ~ log p(x,h) - log q(h|x)
        log_nll = self.joint_model.log_joint_prob(
            x, h
        ) - self.proposal_model.log_conditional_prob(h, x)

        return log_nll.detach().cpu().numpy()

    def validation_step(self, batch, batch_idx):
        x, _, idx = batch  # x: [B, D], idx: [B,]
        idx = idx.tolist()

        nll = -self.get_nll(x, idx=idx)

        self.log("valid/nll", nll.mean(), prog_bar=True, sync_dist=True)

        if batch_idx == 0:
            self.validation_step_outputs.append(x[:16])

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

    def state_dict(self):
        state = super().state_dict()
        # include sampler state
        sampler_state = self.sampler.state_dict()
        state["sampler_state"] = sampler_state
        return state

    def load_state_dict(self, state_dict, strict=True):
        # load sampler state
        if "sampler_state" in state_dict:
            sampler_state = state_dict.pop("sampler_state")
            self.sampler.load_state_dict(sampler_state)
        super().load_state_dict(state_dict, strict=strict)

    def on_train_epoch_start(self):
        if self.current_epoch >= self.cache_start_epoch:
            self.sampler.use_cache = True
        else:
            self.sampler.use_cache = False

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
        if "sampler_state" in checkpoint and getattr(self.sampler, "use_cache", False):
            self.sampler.load_state_dict(checkpoint["sampler_state"])
            # after load, ensure sampler cache is on correct device
            self.sampler.to(self.device)
            # broadcast loaded cache to all ranks to be safe
            if dist.is_available() and dist.is_initialized():
                # we expect that checkpoint was saved from rank0 and all ranks loaded same dict,
                # but ensure everyone has the same in distributed environment
                self.sampler.sync_cache()
