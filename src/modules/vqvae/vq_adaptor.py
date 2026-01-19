# src/modules/vqvae/vq_adaptor.py

import torch
from torch import nn
from src.modules.networks import Encoder, Decoder

class ConvEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        ch,
        out_ch,
        ch_mult,
        num_res_blocks,
        attn_resolutions,
        dropout,
        activation,
        norm_type,
        resamp_with_conv,
        resolution,
        z_channels,
        double_z=False,
        **ignore_kwargs,
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels=in_channels,
            ch=ch,
            out_ch=out_ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            activation=activation,
            norm_type=norm_type,
            resamp_with_conv=resamp_with_conv,
            resolution=resolution,
            z_channels=z_channels,
            double_z=double_z,
        )

    def forward(self, x):
        z = self.encoder(x)
        return z
    
    def get_last_layer_weight(self):
        return self.encoder.conv_out.weight
    
class ConvDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        ch,
        out_ch,
        ch_mult,
        num_res_blocks,
        attn_resolutions,
        dropout,
        activation,
        norm_type,
        resamp_with_conv,
        resolution,
        z_channels,
        final_activation="sigmoid",
        **ignore_kwargs,
    ):
        super().__init__()
        self.decoder = Decoder(
            in_channels=in_channels,
            ch=ch,
            out_ch=out_ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            activation=activation,
            norm_type=norm_type,
            resamp_with_conv=resamp_with_conv,
            resolution=resolution,
            z_channels=z_channels,
        )
        self.final_activation = final_activation

    def forward(self, z):
        x_recon = self.decoder(z)
        if self.final_activation == "sigmoid":
            x_recon = torch.sigmoid(x_recon)
        elif self.final_activation == "tanh":
            x_recon = torch.tanh(x_recon)
        elif self.final_activation is None:
            pass
        return x_recon
    
    def get_last_layer_weight(self):
        return self.decoder.conv_out.weight