import torch
import torch.distributed as dist
from lightning.pytorch.callbacks import Callback

from src.utils.codebook_utils import plot_codebook_usage_distribution


class CodebookStats(Callback):
    """
    Unified callback for codebook utilization statistics.

    Requirements on LightningModule:
        - get_codebook_indices(batch) -> 1D LongTensor
        - get_codebook_size() -> int
    """

    def __init__(
        self,
        stage: str = "val",  # "val" or "test"
        log_prefix: str | None = None,
        plot: bool = True,
        sort_by_counter: bool = True,
        use_log_scale: bool = False,
    ):
        assert stage in ["val", "test"]
        self.stage = stage
        self.log_prefix = log_prefix or stage
        self.plot = plot
        self.sort_by_counter = sort_by_counter
        self.use_log_scale = use_log_scale

        # runtime states
        self.codebook_size = None
        self.counter = None

    # -------------------------
    # lifecycle helpers
    # -------------------------
    def _is_distributed(self):
        return dist.is_available() and dist.is_initialized()

    def _sync_counter(self):
        if self._is_distributed():
            dist.all_reduce(self.counter, op=dist.ReduceOp.SUM)

    # -------------------------
    # hooks
    # -------------------------
    def on_validation_start(self, trainer, pl_module):
        if self.stage != "val":
            return
        self._reset_counter(pl_module)

    def on_test_start(self, trainer, pl_module):
        if self.stage != "test":
            return
        self._reset_counter(pl_module)

    def _reset_counter(self, pl_module):
        if not hasattr(pl_module, "get_codebook_size"):
            raise RuntimeError(
                "LightningModule must implement get_codebook_size() "
                "to use CodebookStatsCallback."
            )

        self.codebook_size = int(pl_module.get_codebook_size())
        self.counter = torch.zeros(
            self.codebook_size,
            dtype=torch.long,
            device=pl_module.device,
        )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.stage != "val":
            return
        self._accumulate(pl_module, batch)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.stage != "test":
            return
        self._accumulate(pl_module, batch)

    @torch.no_grad()
    def _accumulate(self, pl_module, batch):
        if not hasattr(pl_module, "get_codebook_indices"):
            raise RuntimeError(
                "LightningModule must implement get_codebook_indices(batch) "
                "to use CodebookStatsCallback."
            )

        indices = pl_module.get_codebook_indices(batch)

        if indices.dtype != torch.long:
            indices = indices.long()

        self.counter.index_add_(
            0,
            indices,
            torch.ones_like(indices, dtype=torch.long),
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.stage != "val":
            return
        self._finalize(pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        if self.stage != "test":
            return
        self._finalize(pl_module)

    # -------------------------
    # finalize & logging
    # -------------------------
    def _finalize(self, pl_module):
        self._sync_counter()

        used_codewords = torch.sum(self.counter > 0).item()
        utilization_rate = used_codewords / self.codebook_size * 100.0

        # scalar logging
        pl_module.log(
            f"{self.log_prefix}/codebook_utilization",
            utilization_rate,
            prog_bar=True,
            sync_dist=True,
        )
        pl_module.log(
            f"{self.log_prefix}/used_codewords",
            used_codewords,
            prog_bar=False,
            sync_dist=True,
        )

        # plot distribution
        if self.plot and hasattr(pl_module.logger, "experiment"):
            fig_dict, _ = plot_codebook_usage_distribution(
                codebook_counter=self.counter.cpu().numpy(),
                codebook_size=self.codebook_size,
                tag_prefix=self.log_prefix,
                save_to_disk=False,
                sort_by_counter=self.sort_by_counter,
                use_log_scale=self.use_log_scale,
            )

            for tag, fig in fig_dict.items():
                pl_module.logger.experiment.add_figure(
                    tag,
                    fig,
                    pl_module.current_epoch,
                )

        # reset for safety
        self.counter.zero_()
