# src/models/latent_transformer.py
"""This module defines the LatentTransformer model, which is a transformer-based architecture designed for processing latent representations of images.
The model consists of an encoder that maps input images to a latent space, and a transformer that operates on the latent representations to capture complex dependencies.
The module also includes utilities for training and evaluating the model, as well as for visualizing the learned latent space.

The LatentTransformer model is particularly useful for tasks such as image generation, reconstruction, and representation learning, where capturing the underlying structure of the data in a compact latent space is crucial.
"""

from lightning.pytorch import LightningModule
import torch.nn as nn
import torch
import logging

from src.modules.permuter import AbstractPermuter
from src.modules.gpts.mingpt import GPT
from src.base.base_dataset import JsaDataset
from src.utils.module_utils import (
    load_from_file,
    instantiate_from_config,
    load_from_config,
)

logger = logging.getLogger(__name__)


class LatentTransformer(LightningModule):
    def __init__(
        self,
        first_stage_config: dict,
        cond_stage_config: dict,
        transformer: GPT,
        permuter: AbstractPermuter,
        pkeep: float = 0.9,
        cond_stage_dataset_key: str = "image",
    ):
        super().__init__()

        self.first_stage_model = self.init_first_stage_from_config(first_stage_config)
        self.cond_stage_model = self.init_cond_stage_from_config(cond_stage_config)

        self.transformer: GPT = transformer
        self.permuter: AbstractPermuter = permuter

        self.pkeep = pkeep
        self.cond_stage_dataset_key = cond_stage_dataset_key

    ############################# Load models ##########################################################

    def _load_model_from_config(self, config, stage_prefix=""):
        """
        Helper method to load a model based on the configuration dict.
        It checks for 'config_path' to load from file, otherwise loads from the config dict itself.
        """
        if not config:
            return None

        prefix = f"[{stage_prefix} stage] " if stage_prefix else ""

        try:
            config_path = config.get("config_path")
            ckpt_path = config.get("ckpt_path")

            if config_path:
                logger.info(f"{prefix}Loading model from file: {config_path}")
                if ckpt_path:
                    logger.info(f"{prefix}Restoring weights from: {ckpt_path}")
                return load_from_file(config_path, ckpt_path, freeze=True)
            else:
                cls_path = config.get("class_path", "Unknown Class")
                logger.info(
                    f"{prefix}Loading model from config dict (class: {cls_path})"
                )
                if ckpt_path:
                    logger.info(f"{prefix}Restoring weights from: {ckpt_path}")
                return load_from_config(config, ckpt_path, freeze=True)
        except Exception as e:
            raise RuntimeError(f"{prefix}Error loading model: {e}")

    def init_first_stage_from_config(self, config):
        return self._load_model_from_config(config, stage_prefix="First Stage")

    def init_cond_stage_from_config(self, config):
        mode = config.get("mode", "first_stage_model")

        if mode == "first_stage_model":
            cond_model = self.first_stage_model

            # Ensure the dataset key is set to image for first stage model
            self.cond_stage_dataset_key = JsaDataset.IMAGE_KEY

        elif mode == "external_model":
            cond_model = self._load_model_from_config(config, stage_prefix="Cond Stage")
        elif mode == "no_cond":
            cond_model = instantiate_from_config(config)

            # Default to image key even if no condition model is used,
            # as the model will generate its own condition tokens (e.g., SOS token)
            self.cond_stage_dataset_key = JsaDataset.IMAGE_KEY
        else:
            raise ValueError(
                f"Invalid mode '{mode}' for cond stage model. Expected 'first_stage_model', 'external_model', or 'no_cond'."
            )

        return cond_model

    ################################## Training methods ##########################################################
    def forward(self, x, cond):

        # Ensure that the condition input is not None
        # Even for unconditional training, we will generate a default condition token (e.g., SOS token) in the cond_stage_model,
        # so cond should never be None when passed to this method.
        assert (
            cond is not None
        ), "Condition input cannot be None. Please provide a valid condition input."

        latent_token = self.encode_to_latent(x)  # [B, T_latent]

        # [B, T_cond] or [B, 1] if using SOS token for unconditional training
        cond_token = self.encode_cond(cond)

        # Assuming we want to predict the latent tokens themselves
        target_tokens = latent_token  # [B, T_latent]

        if self.training and self.pkeep < 1.0:
            # Apply token dropping for regularization during training
            mask = torch.bernoulli(
                self.pkeep * torch.ones(latent_token.shape, device=latent_token.device)
            )
            mask = mask.bool()
            random_tokens = torch.randint_like(
                latent_token, low=0, high=self.transformer.config.vocab_size
            )
            input_token = torch.where(mask, latent_token, random_tokens)
        else:
            input_token = latent_token

        # Concatenate condition tokens if available
        # Shape of inputs to transformer: [B, T_cond + T_latent]
        inputs = torch.cat([cond_token, input_token], dim=1)

        # Predict next token in the sequence
        # Return shape of logits: [B, T_latent+T_cond-1, vocab_size]
        logits, _ = self.transformer(inputs[:, :-1])

        # Focus on the latent token predictions
        logits = logits[
            :,
            -latent_token.shape[1] :,
        ]  # [B, T_latent, vocab_size], predicting the next token in the latent sequence
        return logits, target_tokens

    def get_input(self, batch, key):
        # Safe method to retrieve input data from the batch with error handling
        if key not in batch:
            raise KeyError(
                f"Key '{key}' not found in batch. Available keys: {list(batch.keys())}"
            )
        return batch[key]

    def _shared_step(self, batch, batch_idx):
        x = self.get_input(batch, JsaDataset.IMAGE_KEY)
        cond = self.get_input(batch, self.cond_stage_dataset_key)

        logits, target_tokens = self.forward(x, cond)
        loss = nn.CrossEntropyLoss()(
            logits.view(-1, logits.size(-1)), target_tokens.view(-1)
        )

        return loss

    def training_step(self, batch, batch_idx):
        # Implement the training logic here
        loss = self._shared_step(batch, batch_idx)
        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    @torch.no_grad()
    def encode_to_latent(self, x):
        # Implement the logic to encode input images to latent representations here
        pass

    @torch.no_grad()
    def encode_cond(self, cond_input):
        # Implement the logic to encode conditional input data here
        pass

    ################################# Evaluation and visualization methods ##########################################################
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # Implement the validation logic here
        loss = self._shared_step(batch, batch_idx)
        self.log(
            "valid/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def log_images(self, imgs, step):
        # Implement the logic to log images for visualization here
        pass

    ################################### Optimizer and scheduler configuration ##########################################################
    def configure_optimizers(self):
        # Implement the logic to configure optimizers and learning rate schedulers here
        pass
