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
from src.utils.file_logger import get_file_logger

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
        base_learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
    ):
        super().__init__()

        self.first_stage_model = self.init_first_stage_from_config(first_stage_config)
        self.cond_stage_model = self.init_cond_stage_from_config(cond_stage_config)

        self.transformer: GPT = transformer
        self.permuter: AbstractPermuter = permuter

        self.pkeep = pkeep
        self.cond_stage_dataset_key = cond_stage_dataset_key
        self.learning_rate = base_learning_rate
        self.weight_decay = weight_decay

        self.train_logger = (
            None  # will be initialized in on_fit_start() when trainer is available
        )

        # For gradient logging callback
        self.grad_norm_modules = {
            "transformer": self.transformer,
        }

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
    def on_fit_start(self):

        log_dir = self.trainer.log_dir if self.trainer else "logs"
        self.train_logger = get_file_logger(
            log_path=f"{log_dir}/training.log",
            name="train_logger",
            rank=self.trainer.global_rank if self.trainer else 0,
        )

    def on_train_start(self):

        if self.train_logger is not None:
            self.train_logger.info("Starting training...")
            self.train_logger.info(
                f"Transformer Model Config: {self.transformer.config}"
            )
            self.train_logger.info(
                f"First Stage Model: {type(self.first_stage_model).__name__}"
            )
            self.train_logger.info(
                f"Conditional Stage Model: {type(self.cond_stage_model).__name__}"
            )
            self.train_logger.info(
                f"Permutation Strategy: {type(self.permuter).__name__}"
            )
            self.train_logger.info(f"Transformer Model Structure: {self.transformer}")

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
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        return loss

    @torch.no_grad()
    def encode_to_latent(self, x):
        # Implement the logic to encode input images to latent representations here
        if hasattr(self.first_stage_model, "tokenize") and callable(
            self.first_stage_model.tokenize
        ):
            indices = self.first_stage_model.tokenize(
                x, flatten=True
            )  # [B, C, H, W] -> [B, H'*W']
            indices = self.permuter(indices)
            return indices
        else:
            raise NotImplementedError(
                "The first stage model must implement a 'tokenize' method to encode images to latent tokens."
            )

    @torch.no_grad()
    def encode_cond(self, cond_input):
        # Implement the logic to encode conditional input data here
        if hasattr(self.cond_stage_model, "tokenize") and callable(
            self.cond_stage_model.tokenize
        ):
            cond_token = self.cond_stage_model.tokenize(
                cond_input, flatten=True
            )  # [B, C, H, W] -> [B, H', W']

            return cond_token
        elif hasattr(self.cond_stage_model, "encode") and callable(
            self.cond_stage_model.encode
        ):
            # For models like SOSProvider that generate condition tokens based on the input batch, we can call the encode method directly.
            cond_token = self.cond_stage_model.encode(
                cond_input
            )  # [B, C, H, W] -> [B, T_cond]
            return cond_token
        else:
            raise NotImplementedError(
                "The conditional stage model must implement either a 'tokenize' or 'encode' method."
            )

    @torch.no_grad()
    def decode_from_latent(self, latent_tokens):
        # latent_tokens: [B, T_latent]
        # Implement the logic to decode latent tokens back to images here
        if hasattr(self.first_stage_model, "detokenize") and callable(
            self.first_stage_model.detokenize
        ):
            # If the first stage model has a decode method, we can use it to reconstruct images from latent tokens.
            # We may need to reverse the permutation before decoding.
            permuted_tokens = self.permuter(
                latent_tokens, reverse=True
            )  # [B, T_latent] -> [B, T_latent]

            #! Important note:
            # When in the early stage durning unconditional training, the generated latent tokens may not be meaningful, so it will predict `sos_token`, which does not appears in the first stage model's codebook, and thus cannot be decoded by the first stage model, which may cause errors or produce garbage reconstructions.
            # To handle this, we replace these out-of-vocabulary tokens with a valid token (e.g., the token for all-zero image) before decoding.
            # There is no need to worry about this correction, because in the late phase of training, the model will learn to generate valid tokens.
            sos_token = self.cond_stage_model.sos_token if hasattr(self.cond_stage_model, "sos_token") else None
            if sos_token is not None:
                valid_token = 0  # Assuming token 0 corresponds to a valid image (e.g., all-zero image) in the first stage model's codebook
                permuted_tokens = torch.where(
                    permuted_tokens == sos_token, valid_token, permuted_tokens
                )
            
            # [B, T_latent -> [B, C, H, W]
            # only for square T_latent, we can infer H and W as sqrt(T_latent)
            # otherwise, we need to provide `shape`, like:
            # self.first_stage_model.decode(permuted_tokens, shape=(H, W))
            reconstructed_images = self.first_stage_model.detokenize(permuted_tokens)
            return reconstructed_images
        else:
            raise NotImplementedError(
                "The first stage model must implement a 'detokenize' method to reconstruct images from latent tokens."
            )

    ################################# Evaluation and visualization methods ##########################################################
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # Implement the validation logic here
        loss = self._shared_step(batch, batch_idx)
        self.log(
            "valid/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        return loss

    def log_images(
        self,
        batch,
        temperature=1.0,
        top_k=100,
        top_p=1.0,
        callback=lambda k: None,
        **kwargs,
    ):
        """Implement the logic to generate and log images during validation here. This method will be called by the ImageLoggerCallback at specified intervals during training and validation.

        Args:
            batch: the input batch from the dataloader, which can be used to extract any necessary information for conditioning the generation (e.g., class labels, input images for reconstruction, etc.)
            temperature: float - the sampling temperature to control randomness during generation
            top_k: int or None - if specified, only consider the top_k most likely tokens at each step for sampling
            top_p: float - if specified, only consider the smallest set of tokens whose cumulative probability exceeds top_p for sampling
            callback: function - a callback function that will be called at each generation step with the current step index, useful for logging or early stopping during generation
        Returns:
            generated_images: a dictionary of images to be logged, where keys are the names of the image groups (e.g., "reconstructions", "samples", etc.) and values are tensors of shape [N, C, H, W] containing the images to be logged.

                These keys now contains:
                    - inputs: original input images from the batch
                    - reconstructions: images reconstructed from true latent tokens,
                        should be the best reconstructions we can get from the model,
                        useful for monitoring reconstruction quality during training
                    - samples_from_scratch: images generated by sampling from the model without any conditioning (or with default conditioning like SOS token),
                        useful for monitoring the generative quality of the model during training
                    - samples_from_half: images generated by sampling from the model starting with the first half of the true latent tokens and then sampling the rest,
                        useful for monitoring how well the model can continue a given latent sequence during training
                    - samples_deterministic: images generated by taking the argmax at each step instead of sampling,
                        useful for monitoring the best possible generations from the model during training
        """

        log = dict()
        # Get original input images for reference
        input_images = self.get_input(batch, JsaDataset.IMAGE_KEY)
        cond_images = self.get_input(batch, self.cond_stage_dataset_key)

        log["inputs"] = input_images

        # Get true latent tokens and condition tokens
        true_latent_tokens = self.encode_to_latent(input_images)  # [B, T_latent]
        cond_tokens = self.encode_cond(cond_images)  # [B, T_cond]

        sampling_max_length = true_latent_tokens.shape[
            1
        ]  # we can set max_length to the length of true latent tokens for fair comparison

        # Generate reconstructions from true latent tokens (no sampling, just decode)
        reconstructions = self.decode_from_latent(true_latent_tokens)  # [B, C, H, W]
        log["reconstructions"] = reconstructions

        # Generate samples from scratch (using only condition tokens, no latent tokens)
        samples_from_scratch = self.sample(
            latent_tokens=torch.empty(
                (input_images.shape[0], 0), dtype=torch.long, device=input_images.device
            ),  # empty latent tokens
            cond_token=cond_tokens,  # use condition tokens for sampling (can be SOS token for unconditional generation)
            max_length=sampling_max_length,  # generate the same number of tokens as the true latent tokens
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            callback=callback,
        )  # [B, C, H, W]
        log["samples_from_scratch"] = samples_from_scratch

        # Generate samples starting from the first half of the true latent tokens
        half_length = true_latent_tokens.shape[1] // 2
        samples_from_half = self.sample(
            latent_tokens=true_latent_tokens[
                :, :half_length
            ],  # use the first half of the true latent tokens as the starting point
            cond_token=cond_tokens,  # use condition tokens for sampling
            max_length=sampling_max_length,  # generate the same total number of tokens as the true latent tokens
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            callback=callback,
        )  # [B, C, H, W]
        log["samples_from_half"] = samples_from_half

        # Generate deterministic samples by taking the argmax at each step instead of sampling
        samples_deterministic = self.sample(
            latent_tokens=torch.empty(
                (input_images.shape[0], 0), dtype=torch.long, device=input_images.device
            ),  # empty latent tokens
            cond_token=cond_tokens,  # use condition tokens for sampling
            max_length=sampling_max_length,  # generate the same number of tokens as the true latent tokens
            temperature=0.0,  # set temperature to 0 for deterministic sampling (argmax)
            top_k=None,  # disable top_k filtering for deterministic sampling
            top_p=1.0,  # disable top_p filtering for deterministic sampling
            callback=callback,
        )  # [B, C, H, W]
        log["samples_deterministic"] = samples_deterministic

        return log

    ################################## Sampling and generation methods ##########################################################
    @torch.no_grad()
    def sample(
        self,
        latent_tokens,
        cond_token=None,
        max_length=512,
        temperature=1.0,
        top_k=None,
        top_p=1.0,
        callback=lambda k: None,
    ):
        """Implement the logic to generate new images by sampling from the model here

        Args:
            latent_tokens: [B, T_pre] - the initial latent tokens to start generation from (can be empty for pure generation)
            cond_token: [B, T_cond] or None
            max_length: int - the maximum length of the generated sequence (including the initial latent tokens)
            temperature: float - the sampling temperature to control randomness
            top_k: int or None - if specified, only consider the top_k most likely tokens at each step for sampling
            top_p: float - if specified, only consider the smallest set of tokens whose cumulative probability exceeds top_p for sampling
            callback: function - a callback function that will be called at each generation step with the current step index, useful for logging or early stopping during
                generation
        Returns:
            generated_images: [B, C, H, W] - the generated images decoded from the generated latent tokens
        """

        assert (
            cond_token is not None
        ), "Condition token cannot be None for sampling. Please provide a valid condition token (e.g., SOS token for unconditional generation)."
        inputs = torch.cat([cond_token, latent_tokens], dim=1)  # [B, T_cond + T_pre]

        assert (
            not self.transformer.training
        ), "Model must be in evaluation mode for sampling. Please call model.eval() before sampling."

        # For we will cut off the condition tokens, so the length conditional tokens is not counted in max_length,
        # but we need to ensure that the initial latent tokens (T_pre) plus the generated tokens (steps) do not exceed max_length.
        current_valid_length = latent_tokens.shape[1]  # T_pre
        assert (
            current_valid_length < max_length
        ), f"Initial input length {current_valid_length} must be less than max_length {max_length} for sampling."
        steps = max_length - current_valid_length

        generated_tokens = self.transformer.sample(
            inputs,
            steps=steps,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            callback=callback,
        )  # [B, T_cond + max_length]

        # We only care about the generated latent tokens (the part after the condition tokens)
        generated_latent_tokens = generated_tokens[
            :, cond_token.shape[1] :
        ]  # [B, T_cond + max_length] -> [B, max_length]

        # We should verify that the generated_latent_tokens have the expected shape (e.g., [B, T_latent]) before decoding, and handle cases where the generation may produce more tokens than expected.
        generated_images = self.decode_from_latent(generated_latent_tokens)
        return generated_images

    ################################### Optimizer and scheduler configuration ##########################################################
    def configure_optimizers(self):
        """Implement the logic to configure optimizers and learning rate schedulers here"""

        # Learning rate
        batch_size = (
            self.trainer.datamodule.batch_size
            if self.trainer and self.trainer.datamodule
            else 1
        )
        num_devices = self.trainer.num_devices if self.trainer else 1
        accumulate_grad_batches = (
            self.trainer.accumulate_grad_batches if self.trainer else 1
        )
        effective_batch_size = batch_size * num_devices * accumulate_grad_batches
        self.learning_rate = (
            self.learning_rate * effective_batch_size
        )  # scale learning rate based on effective batch size

        if self.train_logger is not None:
            self.train_logger.info(
                f"Configuring optimizers with effective batch size {effective_batch_size}, learning_rate {self.learning_rate}"
            )
        else:
            print(
                f"Configuring optimizers with effective batch size {effective_batch_size}, learning_rate {self.learning_rate}"
            )

        # Using AdamW optimizer for the transformer parameters
        # for different modules, using different decay rates can be beneficial (e.g., no decay for bias and LayerNorm weights)
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay_params = []
        no_decay_params = []

        # Modules that should have weight decay applied to their parameters (e.g., Linear layers)
        whitelist_weight_modules = (nn.Linear,)
        # Modules that should not have weight decay applied to their parameters (e.g., LayerNorm, Embedding)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

        # Create a map from parameter name to its parent module
        # Note: This map is needed because named_parameters() doesn't give module info directly
        name_to_module = {n: m for n, m in self.transformer.named_modules()}

        for name, param in self.transformer.named_parameters():
            if not param.requires_grad:
                continue  # skip frozen parameters

            # Positional embeddings interact with weight decay very weirdly, so we turn it off
            if name.endswith("pos_emb"):
                no_decay_params.append(param)
                continue

            # All bias are not decayed
            if name.endswith("bias"):
                no_decay_params.append(param)
                continue

            # Handle weights based on module type
            # We need to find the direct parent module of the parameter
            # e.g., 'blocks.0.ln1.weight' -> parent is the LayerNorm module 'blocks.0.ln1'
            parent_name = ".".join(name.split(".")[:-1]) if "." in name else ""
            parent_module = name_to_module.get(parent_name, None)

            if isinstance(parent_module, blacklist_weight_modules):
                no_decay_params.append(param)
            elif isinstance(parent_module, whitelist_weight_modules) and name.endswith(
                "weight"
            ):
                decay_params.append(param)
            else:
                # Default behavior for any other parameters (usually no decay is safer if unsure,
                # or you can inspect them. The original code crashed if not covered,
                # but most params fall into above buckets).
                # Assuming original logic: if it's weight and whitelist, decay; else no decay.
                # To be strictly safe and match original "union check":
                no_decay_params.append(param)

        # Create optimizer object
        optim_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.learning_rate, betas=(0.9, 0.95)
        )
        return optimizer
