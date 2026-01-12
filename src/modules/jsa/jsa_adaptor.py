# src/modules/jsa/jsa_adaptor.py
# For JSA model: MLP encoder and decoder
from torch import nn
from src.modules.networks import MLPNetwork
import torch
import math
from src.modules.networks import Encoder, Decoder


class MLPEncoder(nn.Module):
    def __init__(
        self,
        mlp_args,
    ):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = MLPNetwork(**mlp_args)

    def forward(self, x):
        x = self.flatten(x)
        feature = self.net(x)
        return feature


class MLPDecoder(nn.Module):
    def __init__(
        self,
        mlp_args,
        output_shape=(1, 28, 28),
    ):
        super().__init__()
        self.net = MLPNetwork(**mlp_args)
        self.unflatten = nn.Unflatten(1, output_shape)

    def forward(self, x):
        feature = self.net(x)
        feature = self.unflatten(feature)
        return feature  # return to original shape


class ConvDecoder(nn.Module):
    def __init__(
        self,
        num_categories,
        num_latent_vars,
        embedding_dims,
        ch,
        out_ch,
        ch_mult,
        num_res_blocks,
        attn_resolutions,
        dropout,
        activation,
        norm_type,
        resamp_with_conv,
        in_channels,
        resolution,
        z_channels,
        final_activation="sigmoid",
        **ignore_kwargs,
    ):
        super().__init__()
        if len(num_categories) == 1 and num_latent_vars > 1:
            self._num_categories = list(num_categories) * num_latent_vars
        else:
            assert (
                len(num_categories) == num_latent_vars
            ), "num_categories must be an integer or a list of length num_latent_vars"
            self._num_categories = list(num_categories)

        # Add Embedding layer for Categorical latent variables
        if embedding_dims is None:
            self.embedding_dims = [
                min(2, int(math.log2(K))) for K in self._num_categories
            ]
        elif len(embedding_dims) == 1 and num_latent_vars > 1:
            self.embedding_dims = list(embedding_dims) * num_latent_vars
        else:
            assert (
                len(embedding_dims) == num_latent_vars
            ), "embedding_dims must be an integer or a list of length num_latent_vars"
            self.embedding_dims = list(embedding_dims)

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=K, embedding_dim=emb_dim)
                for K, emb_dim in zip(self._num_categories, self.embedding_dims)
            ]
        )
        
        self.decoder = Decoder(
            ch=ch,
            out_ch=out_ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            activation=activation,
            norm_type=norm_type,
            resamp_with_conv=resamp_with_conv,
            in_channels=in_channels,
            resolution=resolution,
            z_channels=z_channels,
        )

        self.final_activation = final_activation
        
        self._latent_dim = sum(self.embedding_dims)
        
    @property
    def latent_dim(self):
        return self._latent_dim
    
    @property
    def num_categories(self):
        return self._num_categories

    def get_last_layer_weight(self):
        return self.decoder.conv_out.weight

    def forward(self, h):
        # h: [N, H, W, num_latent_vars]
        # return x: [N, out_ch, H, W]
        h_embedded = [
            embedding(h[..., i].long())
            for i, embedding in enumerate(self.embeddings)
        ]
        h_densed = torch.cat(h_embedded, dim=-1)  # [N, H, W, sum(emb_dim)]
        h = h_densed.permute(
            0, 3, 1, 2
        ).contiguous()  # [B, H, W, z_channels] -> [B, z_channels, H, W]
        x = self.decoder(h)  # [B, out_ch, H, W]

        # Rescale output to [0, 1] range if needed
        if self.final_activation == "sigmoid":
            x = torch.sigmoid(x)
        elif self.final_activation == "tanh":
            x = torch.tanh(x)
        elif self.final_activation is None:
            pass

        return x


class ConvEncoder(nn.Module):
    def __init__(
        self,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        attn_resolutions=[],
        activation="relu",
        norm_type="group",
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=1,
        resolution=28,
        z_channels=4,
        **ignore_kwargs,
    ):
        super().__init__()
        self.encoder = Encoder(
            ch=ch,
            out_ch=out_ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            activation=activation,
            norm_type=norm_type,
            dropout=dropout,
            resamp_with_conv=resamp_with_conv,
            in_channels=in_channels,
            resolution=resolution,
            z_channels=z_channels,
            double_z=False,
        )
        self.last_proj = torch.nn.Conv2d(
            z_channels, out_ch, kernel_size=1, stride=1, padding=0
        )

    def get_last_layer_weight(self):
        return self.encoder.conv_out.weight

    def forward(self, x):
        # x: [B, in_channels, H, W]
        # return h: [B, H, W, out_ch]
        x = self.encoder(x)  # [B, z_channels, H, W]
        x = self.last_proj(x)  # [B, out_ch, H, W]
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, z_channels]
        return x
