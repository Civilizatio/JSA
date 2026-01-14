import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from src.modules.networks import Encoder, Decoder
from src.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from src.modules.losses.vqperceptual import VQLPIPSWithDiscriminator
from src.utils.file_logger import get_file_logger

import torch.distributed as dist
from src.utils.codebook_utils import plot_codebook_usage_distribution

class VQModel(LightningModule):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        loss: VQLPIPSWithDiscriminator,
        quantizer: VectorQuantizer,
        base_learning_rate=2e-4,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
    ):
        super().__init__()
        self.image_key = image_key
        self.encoder: Encoder = encoder
        self.decoder: Decoder = decoder
        self.loss: VQLPIPSWithDiscriminator = loss
        self.quantizer: VectorQuantizer = quantizer
        self.base_learning_rate = base_learning_rate
        self.learning_rate = base_learning_rate

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        self.train_logger = None
        self.automatic_optimization = False
        
        self.log_codebook_utilization_valid = True
        self.log_codebook_utilization_test = False
        
        self.grad_norm_modules = {
            "encoder": self.encoder,
            "decoder": self.decoder,
            "quantizer": self.quantizer,
        }

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        quant, emb_loss, info = self.quantizer(h)
        return quant, emb_loss, info

    def decode(self, quant):
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        shape = None
        if len(code_b.shape) == 3:
            b, h, w = code_b.shape
            shape = (b, h, w, self.quantizer.e_dim)

        quant_b = self.quantizer.get_codebook_entry(code_b, shape=shape)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        # if len(x.shape) == 3:
        #     x = x[..., None]
        # x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        x = x.to(memory_format=torch.contiguous_format)
        return x.float()

    def on_fit_start(self):
        log_dir = self.logger.log_dir if self.logger else "logs"
        self.train_logger = get_file_logger(
            log_dir + "/train.log",
            name="train_logger",
            rank=self.global_rank if hasattr(self, "global_rank") else 0,
        )

    def on_train_start(self):
        if self.train_logger is not None:
            self.train_logger.info(f"Model:\n{self}")
            
    
    def training_step(self, batch, batch_idx):
        # Get optimizers
        opt_ae, opt_disc = self.optimizers()

        # Forward pass
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)

        # Get losses and logs from the loss function
        aeloss, discloss, log_dict_ae, log_dict_disc = self.loss(
            qloss,
            x,
            xrec,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
        )

        # Log dictionaries
        batch_size = x.shape[0]
        self.log_dict(
            log_dict_ae,
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log_dict(
            log_dict_disc,
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        # --- Optimize Autoencoder (Generator) ---
        opt_ae.zero_grad()
        self.manual_backward(aeloss)
        opt_ae.step()

        # --- Optimize Discriminator ---
        opt_disc.zero_grad()
        self.manual_backward(discloss)
        opt_disc.step()

    
    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        quant, qloss, info = self.encode(x)
        xrec = self.decode(quant)
        

        # The loss function now returns multiple values, we only need the logs for validation
        _, _, log_dict_ae, log_dict_disc = self.loss(
            qloss,
            x,
            xrec,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="valid",
        )

        batch_size = x.shape[0]
        self.log_dict(
            log_dict_ae,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log_dict(
            log_dict_disc,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        return self.log_dict
    

    def configure_optimizers(self):

        batch_size = (
            self.trainer.datamodule.batch_size
            if self.trainer and self.trainer.datamodule
            else 64
        )
        n_gpus = max(1, self.trainer.num_devices) if self.trainer else 1
        accumulated_batches = (
            self.trainer.accumulate_grad_batches if self.trainer else 1
        )
        effective_batch_size = batch_size * n_gpus * accumulated_batches
        self.learning_rate = self.base_learning_rate * effective_batch_size
        lr = self.learning_rate
        if self.train_logger is not None:
            self.train_logger.info(
                f"Configuring optimizers with effective batch size {effective_batch_size} and learning rate {lr}"
            )
        else:
            print(
                f"Configuring optimizers with effective batch size {effective_batch_size} and learning rate {lr}"
            )

        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quantizer.parameters()),
            lr=lr,
            betas=(0.5, 0.9),
        )
        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9)
        )
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log
    
    def get_codebook_indices(self, batch):
        x = self.get_input(batch, self.image_key)
        _, _, info = self.encode(x)
        return info[2]  # indices
    
    def get_codebook_size(self):
        return self.quantizer.n_e


    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
        return x


class VQNoDiscModel(VQModel):
    def __init__(
        self,
        encoder,
        decoder,
        loss,
        quantizer,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
    ):
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            loss=loss,
            quantizer=quantizer,
            ckpt_path=ckpt_path,
            ignore_keys=ignore_keys,
            image_key=image_key,
            colorize_nlabels=colorize_nlabels,
            monitor=monitor,
        )

        self.automatic_optimization = True

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)

        # 假设 loss 函数在这里仅返回重构损失和日志字典
        # 注意：这里的 self.loss 应该是一个不包含判别器逻辑的 Loss Module (例如 VQLPIPS)
        loss, log_dict = self.loss(qloss, x, xrec, self.global_step, split="train")

        self.log(
            "train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )
        self.log_dict(
            log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)

        loss, log_dict = self.loss(qloss, x, xrec, self.global_step, split="valid")

        self.log(
            "valid/loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True
        )
        self.log_dict(
            log_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True
        )

        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quantizer.parameters()),
            lr=lr,
            betas=(0.5, 0.9),
        )
        return optimizer
