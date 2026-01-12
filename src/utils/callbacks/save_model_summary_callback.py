from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.model_summary import ModelSummary


class LogModelSummary(Callback):
    """
    Log Lightning ModelSummary into existing train.log via file logger.

    Requirements:
        - LightningModule has `train_logger` attribute
          (created in on_fit_start or earlier)
    """

    def __init__(self, max_depth=2, check_eval_trainable=True):
        super().__init__()
        self.max_depth = max_depth
        self.check_eval_trainable = check_eval_trainable

    def on_train_start(self, trainer, pl_module):
        # ---- safety guards ----
        if not trainer.is_global_zero:
            return

        train_logger = getattr(pl_module, "train_logger", None)
        if train_logger is None:
            return

        # ---- model summary ----
        summary = ModelSummary(pl_module, max_depth=self.max_depth)
        summary_str = str(summary)

        train_logger.info("========== Model Summary ==========")
        train_logger.info("\n" + summary_str)
        train_logger.info("===================================")

        # ---- optional: dangerous state check ----
        if self.check_eval_trainable:
            for name, module in pl_module.named_modules():
                if not module.training:
                    for p in module.parameters(recurse=False):
                        if p.requires_grad:
                            train_logger.warning(
                                f"[WARNING] Module `{name}` is in eval() "
                                f"but has trainable parameters."
                            )
