# src/utils/grad_norm_callback.py
import torch
from lightning.pytorch.callbacks import Callback


class GradientNormCallback(Callback):
    """Compute gradient norm for each optimizer separately and log them.

    Works with manual optimization (automatic_optimization = False).
    """

    def __init__(self, norm_type=2, log_every_n_steps=1):
        super().__init__()
        self.norm_type = norm_type
        self.log_every_n_steps = log_every_n_steps

    @torch.no_grad()
    def on_after_backward(self, trainer, pl_module):
        step = trainer.global_step

        if step % self.log_every_n_steps != 0:
            return

        # Compute gradient norm for joint model
        joint_norm = self._compute_grad_norm(pl_module.joint_model, self.norm_type)
        if joint_norm is not None:
            pl_module.log(
                "train/grad_norm_joint",
                joint_norm,
                on_step=True,
                prog_bar=False,
                logger=True,
            )

        # Compute gradient norm for proposal model
        proposal_norm = self._compute_grad_norm(
            pl_module.proposal_model, self.norm_type
        )
        if proposal_norm is not None:
            pl_module.log(
                "train/grad_norm_proposal",
                proposal_norm,
                on_step=True,
                prog_bar=False,
                logger=True,
            )

    def _compute_grad_norm(self, model, norm_type):
        """Compute the total gradient norm of the model parameters."""

        parameters = [p for p in model.parameters() if p.grad is not None]
        if len(parameters) == 0:
            return None
        device = parameters[0].grad.device
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()
