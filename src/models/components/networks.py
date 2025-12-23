# src/models/components/networks.py
# Define some network components here for proposal and joint models

import torch
import torch.nn as nn
import numpy as np
from src.utils.module_utils import get_act_func_by_name


def build_mlp(input_dim, layers, output_dim, activation, final_activation=None):
    """
    Build a multi-layer perceptron (MLP).
    Args:
        input_dim (int): Input dimension.
        layers (list of int): List of hidden layer sizes.
        output_dim (int): Output dimension.
        activation (str): Name of the activation function.
        final_activation (str, optional): Name of the final activation function. Defaults to None.
            - If "sigmoid", output will be in [0, 1]
            - If "tanh", output will be in [-1, 1]
            - If None, no activation on final layer.
    Returns:
        nn.Sequential: The constructed MLP.
    """
    modules = []
    in_dim = input_dim
    for layer_dim in layers:
        modules.append(nn.Linear(in_dim, layer_dim))
        act_func = get_act_func_by_name(activation)
        modules.append(act_func)
        in_dim = layer_dim
    # Last layer without activation
    if final_activation is not None:
        modules.append(nn.Linear(in_dim, output_dim))
        final_act_func = get_act_func_by_name(final_activation)
        modules.append(final_act_func)
    else:
        modules.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*modules)


class MLPNetwork(nn.Module):
    """A simple Multi-Layer Perceptron network"""

    def __init__(
        self,
        input_dim,
        output_dim,
        layers=[512, 512],
        activation="relu",
        final_activation=None,
    ):
        super().__init__()
        self.net = build_mlp(
            input_dim=input_dim,
            layers=layers,
            output_dim=output_dim,
            activation=activation,
            final_activation=final_activation,
        )

    def post_process(self, x):
        """Post-process output x to be in [0, 1] range"""
        if self.net[-1].__class__ == nn.Sigmoid:
            return x  # already in [0, 1]
        elif self.net[-1].__class__ == nn.Tanh:
            return (x + 1) / 2  # scale from [-1, 1] to [0, 1]
        else:
            return torch.clamp(x, 0.0, 1.0)  # clamp to [0, 1]

    def forward(self, x):
        feature = self.net(x)
        return self.post_process(feature)


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


class Normalize(nn.Module):
    def __init__(self, in_channels, norm_type="group", num_groups=32):
        super().__init__()
        if norm_type == "group":
            groups = min(num_groups, in_channels)
            self.norm = torch.nn.GroupNorm(
                num_groups=groups, num_channels=in_channels, eps=1e-6, affine=True
            )
        elif norm_type == "batch":
            self.norm = torch.nn.BatchNorm2d(in_channels, eps=1e-6, affine=True)
        elif norm_type == "instance":
            self.norm = torch.nn.InstanceNorm2d(
                in_channels, eps=1e-6, affine=True, track_running_stats=False
            )
        elif norm_type == "none":
            self.norm = torch.nn.Identity()
        else:
            raise ValueError(f"Unsupported normalization type: {norm_type}")

    def forward(self, x):
        return self.norm(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """Downsample module with optional convolution.

    Only supports downsampling by a factor of 2.
    """

    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        activation="relu",
        dropout,
        temb_channels=512,
        norm_type="group",
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.activation = (
            activation
            if isinstance(activation, nn.Module)
            else get_act_func_by_name(activation)
        )

        self.norm1 = Normalize(in_channels, norm_type=norm_type)
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels, norm_type=norm_type)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = self.activation(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(self.activation(temb))[:, :, None, None]

        h = self.norm2(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels, norm_type="group"):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels, norm_type=norm_type)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1).contiguous()  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1).contiguous()  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        activation="relu",
        norm_type="group",
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        double_z=False,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.activation = get_act_func_by_name(activation)

        # downsampling
        self.conv_in = torch.nn.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        activation=activation,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            activation=activation,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            activation=activation,
        )

        # end
        self.norm_out = Normalize(block_in, norm_type=norm_type)

        # For output logits, we use pointwise convolution,
        # namely, kernel_size=1
        self.conv_out = torch.nn.Conv2d(
            block_in,
            # 2 * z_channels if double_z else z_channels,
            z_channels,
            # For VAE, may use double_z for mean and logvar,
            # here we just output z_channels
            kernel_size=1,
            stride=1,
        )

    def forward(self, x):
        # assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = self.activation(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch: int,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        activation="relu",
        norm_type="group",
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        **ignorekwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.activation = get_act_func_by_name(activation)

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        # print(
        #     "Working with z of shape {} = {} dimensions.".format(
        #         self.z_shape, np.prod(self.z_shape)
        #     )
        # )

        # z to block_in
        # To symmetrically match the Encoder, we use kernel_size=1 here
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=1, stride=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            activation=activation,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            activation=activation,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        activation=activation,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in, norm_type=norm_type)
        self.conv_out = torch.nn.Conv2d(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )

    def forward(self, z):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = self.activation(h)
        h = self.conv_out(h)
        return h


class ConvDecoder(nn.Module):
    def __init__(
        self,
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
        
    def get_last_layer_weight(self):
        return self.decoder.conv_out.weight

    def forward(self, x):
        # x: [B, H, W, z_channels]
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, H, W, z_channels] -> [B, z_channels, H, W]
        x = self.decoder(x) # [B, out_ch, H, W]

        # Rescale output to [0, 1] range if needed
        if self.final_activation == "sigmoid":
            x = torch.sigmoid(x)
        elif self.final_activation == "tanh":
            x = torch.tanh(x)
            x = (x + 1) / 2

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
        )
        
    def get_last_layer_weight(self):
        return self.encoder.conv_out.weight

    def forward(self, x):
        x = self.encoder(x)  # [B, z_channels, H, W]
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, z_channels]
        return x
