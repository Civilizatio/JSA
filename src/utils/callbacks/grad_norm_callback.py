# src/utils/grad_norm_callback.py
import torch
from lightning.pytorch.callbacks import Callback


class GradientNorm(Callback):
    """
    Log gradient norms for arbitrary registered submodules.

    Requirements:
        - pl_module.grad_norm_modules: Dict[str, nn.Module]
        - manual optimization supported
    """

    def __init__(self, norm_type=2.0, log_every_n_steps=1):
        super().__init__()
        self.norm_type = norm_type
        self.log_every_n_steps = log_every_n_steps

    @torch.no_grad()
    def on_after_backward(self, trainer, pl_module):
        step = trainer.global_step
        if step % self.log_every_n_steps != 0:
            return

        module_dict = getattr(pl_module, "grad_norm_modules", None)
        if not isinstance(module_dict, dict):
            return  # silently skip

        for name, module in module_dict.items():
            norm = self._compute_grad_norm(module, self.norm_type)
            if norm is None:
                continue

            pl_module.log(
                f"train/grad_norm/{name}",
                norm,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=False, 
            )

    @staticmethod
    def _compute_grad_norm(module, norm_type):
        params = [p for p in module.parameters() if p.grad is not None]
        if len(params) == 0:
            return None

        device = params[0].grad.device
        norms = torch.stack(
            [torch.norm(p.grad.detach(), norm_type).to(device) for p in params]
        )
        total_norm = torch.norm(norms, norm_type)
        return total_norm.item()
