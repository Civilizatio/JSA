# Net Exporation Documentation

主要对 VQ-VAE，VQ-VAE-2 以及 VQ-GAN 等使用的网络结构进行总结。包括 `Encoder`，`Decoder`，以及 `Discriminator` 等模块的设计细节。

## 1. VQ-VAE

VQ-VAE (Vector Quantized Variational AutoEncoder) 原文: [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) 中并没有给出实现代码。而 Kumar 等人复现了该模型，技术报告和仓库见 [Reproducing "Neural Discrete Representation Learning"](https://github.com/ritheshkumar95/pytorch-vqvae#)。这里总结的也就是 Kumar 等人的实现。

他们在 MNIST 和 CIFAR-10 数据集上复现了 VQ-VAE 的结果，两个数据集上的超参数设置一样。一些超参数（可供我们参考的）列在下面：

| Hyperparameter        | Value|
|----------------------|------|
| Learning Rate         | 3e-4 |
| Batch Size            | 32 (Rather than 128 in the original paper)  |
| Codebook Size        | 512  |
| Embedding Dimension   | 256   |

还发现，在 MNIST 上，使用 Batch Normalization 是必要的。

> 这里的 Embedding Dimension 指的是卷积中间层的特征维度，也就是 `channel` 数。

### 数据预处理

```python
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
``` 


### Encoder

```python
self.encoder = nn.Sequential(
    nn.Conv2d(input_dim, dim, 4, 2, 1),
    nn.BatchNorm2d(dim),
    nn.ReLU(True),
    nn.Conv2d(dim, dim, 4, 2, 1),
    ResBlock(dim),
    ResBlock(dim),
)
```

其中 `ResBlock` 定义为：

```python
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)
```

### Decoder

```python
self.decoder = nn.Sequential(
    ResBlock(dim),
    ResBlock(dim),
    nn.ReLU(True),
    nn.ConvTranspose2d(dim, dim, 4, 2, 1),
    nn.BatchNorm2d(dim),
    nn.ReLU(True),
    nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
    nn.Tanh()
)
```

## VQ-VAE-2

VQ-VAE-2 原文: [Generating Diverse High-Fidelity Images with VQ-VAE-2](https://arxiv.org/abs/1906.00446) 中给出了较为详细的网络结构设计。这里对其进行总结。

VQ-VAE-2 使用了分层的 `Encoder` 和 `Decoder` 结构，分别对应不同的空间分辨率。论文中使用了两个层次，分别称为 `top` 和 `bottom` 层，结构是类似的，只是 `stride` 不同，降采样比例不同。

但是 VQ-VAE-2 没有在 MNIST 或 CIFAR-10 上进行实验，而是在比较大的图像上，例如 ImageNet 和 FFHQ 数据集。这里给出在这两个图像上的一些超参数设置，供参考：

|                              |       ImageNet        |              FFHQ               |
|------------------------------|-----------------------|---------------------------------|
| Input size                   | 256 × 256             | 1024 × 1024                     |
| Latent layers                | 32 × 32, 64 × 64      | 32 × 32, 64 × 64, 128 × 128     |
| beta (commitment loss coefficient) | 0.25                  | 0.25                            |
| Batch size                   | 128                   | 128                             |
| Hidden units                 | 128                   | 128                             |
| Residual units               | 64                    | 64                              |
| Layers                       | 2                     | 2                               |
| Codebook size                | 512                   | 512                             |
| Codebook dimension           | 64                    | 64                              |
| Encoder conv filter size     | 3                     | 3                               |
| Upsampling conv filter size  | 4                     | 4                               |
| Training steps               | 2207444               | 304741                          |



### Encoder

```python
class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)
```

这里的 `ResBlock` 定义如下，发现没有使用 Batch Normalization：

```python
class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out
```

### Decoder

```python
class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)
```

## VQ-GAN

主要的改进是在 VQ-VAE 的基础上加入了 GAN 的训练方式，引入 discriminator 来提升生成图像的质量。

一些超参数（ImageNet 上的设置）列在下面：

| Hyperparameter        | Value|
|----------------------|------|
| Learning Rate         | 4.5e-06 |
| Batch Size            | 256  |
| Codebook Size        | 1024  |
| Embedding Dimension   | 256   |

### Encoder

这里直接是复用 Diffusion Models 中的 Unet 的 Encoder 结构，只不过这里没有 timestep embedding，全部改为 None。


```python
class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        double_z=True,
        **ignore_kwargs
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

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
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
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
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
```

这里的 `ResnetBlock`，`AttnBlock`，`Downsample` 等模块可以参考 Diffusion Models 的实现。

```python
class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
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
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h
```

非线性函数为 `SiLU`：

```python
def nonlinearity(x):
    return x * torch.sigmoid(x)
```

归一化层为 `GroupNorm`：

```python
def Normalize(in_channels):
    return torch.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )
```

按照 ImageNet 的超参数设置：

```yaml
double_z: False
z_channels: 256
resolution: 256
in_channels: 3
out_ch: 3
ch: 128
ch_mult: [ 1,1,2,2,4]  # num_down = len(ch_mult)-1
num_res_blocks: 2
attn_resolutions: [16]
dropout: 0.0
```

则 Encoder 的结构为：不断的下采样，最终将 256x256 的图像降采样到 16x16 的特征图，channel 数从 3 变为 512 (128 * 4)。

### Decoder

Decoder 完全是 Encoder 的镜像结构，不断的上采样，最终将 16x16 的特征图上采样到 256x256 的图像。

```python
class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        **ignorekwargs
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print(
            "Working with z of shape {} = {} dimensions.".format(
                self.z_shape, np.prod(self.z_shape)
            )
        )

        # z to block_in
        self.conv_in = torch.nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
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
        self.norm_out = Normalize(block_in)
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
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
```

### Discriminator

使用 `PatchGAN` 的判别器，与传统的判别器不同的是，不输出单一的标量值，而是输出一个特征图 (feature map)，每个位置的值表示对应图像块的真伪。

一个卷积网络，逐步下采样，最终输出一个特征图。

```python
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
    --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if (
            type(norm_layer) == functools.partial
        ):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)
```

### Total loss

最终定义一个总的损失函数，包含重建损失，GAN 损失，以及 VQ-VAE 的向量量化损失。

其中还有一个 perceptual loss，使用 LPIPS (Learned Perceptual Image Patch Similarity) 作为感知损失。预训练好一个模型，将输入图像和重建图像分别通过该模型，计算它们在特征空间的差异。（这在语音中就是 feature matching loss，distances between the lth feature maps of the kth subdiscriminator.）

```python
class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(
        self,
        disc_start,
        codebook_weight=1.0,
        pixelloss_weight=1.0,
        disc_num_layers=3,
        disc_in_channels=3,
        disc_factor=1.0,
        disc_weight=1.0,
        perceptual_weight=1.0,
        use_actnorm=False,
        disc_conditional=False,
        disc_ndf=64,
        disc_loss="hinge",
    ):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels,
            n_layers=disc_num_layers,
            use_actnorm=use_actnorm,
            ndf=disc_ndf,
        ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(
                nll_loss, self.last_layer[0], retain_graph=True
            )[0]
            g_grads = torch.autograd.grad(
                g_loss, self.last_layer[0], retain_graph=True
            )[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(
        self,
        codebook_loss,
        inputs,
        reconstructions,
        optimizer_idx,
        global_step,
        last_layer=None,
        cond=None,
        split="train",
    ):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(
                inputs.contiguous(), reconstructions.contiguous()
            )
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])

        nll_loss = rec_loss
        # nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        nll_loss = torch.mean(nll_loss)

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(
                    torch.cat((reconstructions.contiguous(), cond), dim=1)
                )
            g_loss = -torch.mean(logits_fake)

            try:
                d_weight = self.calculate_adaptive_weight(
                    nll_loss, g_loss, last_layer=last_layer
                )
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )
            loss = (
                nll_loss
                + d_weight * disc_factor * g_loss
                + self.codebook_weight * codebook_loss.mean()
            )

            log = {
                "{}/total_loss".format(split): loss.clone().detach().mean(),
                "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                "{}/nll_loss".format(split): nll_loss.detach().mean(),
                "{}/rec_loss".format(split): rec_loss.detach().mean(),
                "{}/p_loss".format(split): p_loss.detach().mean(),
                "{}/d_weight".format(split): d_weight.detach(),
                "{}/disc_factor".format(split): torch.tensor(disc_factor),
                "{}/g_loss".format(split): g_loss.detach().mean(),
            }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(
                    torch.cat((inputs.contiguous().detach(), cond), dim=1)
                )
                logits_fake = self.discriminator(
                    torch.cat((reconstructions.contiguous().detach(), cond), dim=1)
                )

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                "{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                "{}/logits_real".format(split): logits_real.detach().mean(),
                "{}/logits_fake".format(split): logits_fake.detach().mean(),
            }
            return d_loss, log
```

## Thoughts

对比而言，在主干网络都是 ResNet 卷积的情况下，VQ-VAE 的设计比较简单，适合在小数据集上进行实验。而 VQ-VAE-2 和 VQ-GAN 的设计更复杂，二者也存在区别。

具体的区别在于上下采样模块的设计，VQ-VAE-2 使用了卷积和转置卷积进行下采样和上采样。而 VQ-GAN 使用 Unet 中的上下采样模块，下采样支持平均池化和卷积，可以选择。而上采样先插值，然后使用卷积进行平滑。最后决定使用 VQ-GAN 的设计。

一个思考的点就是，为什么目前见到的音频使用的上下采样都是使用卷积上下采样，而不是这里的可以选择池化、插值的方法？是因为音频信号的特殊性吗？
感觉确实是图像和音频的差异导致的。

首先对于图像有一个[棋盘格效应（Checkerboard  Artifacts）](#appendix-checkerboard-artifacts)，即如果转置卷积的卷积核大小不能被步长整除，在转置卷积进行的过程中，会出现生成像素点的贡献度周期性不一样的情况，或者说重叠部分不均匀。
对于图像而言，在 *Deconvolution and Checkerboard Artifacts* 中，证明了即使使用 kernel size 能被 stride 整除的转置卷积，也无法完全避免棋盘格效应，因为输入图像本身的统计特性也会影响输出图像的统计特性。因此使用插值加卷积的方法更为稳妥。

对于音频而言，使用插值相当于一个强力的低通滤波器，会将音频的高频细节全部滤掉，而使用转置卷积则可以学习到更复杂的上采样方式，从而保留更多的高频细节。因此在音频处理中，转置卷积可能更适合。

另外一个原因是，音频要求的上采样倍率比图像高很多，通常为 $256\times$ $320\times$ 等，如果使用一步插值加卷积的方式，会导致卷积层看到的局部是完全重复的信号，无法学习到有效的特征。而使用多层插值加卷积，实现上会很复杂。使用转置卷积则可以一步实现高倍率的上采样。以及，音频结构往往具有周期性，转置卷积的大 stride 相当于利用了这种周期性结构，每次复制粘贴一段波形，从而更好地保留音频的周期性特征。

## Design Summary

目前决定使用 VQ-GAN 的设计，一方面是更加现代的，另一方面我比较熟悉 Unet 结构。

代码实现上，Encoder 和 Decoder 直接复用 Diffusion Models 中的 Unet 的 Encoder 和 Decoder 结构，只不过去掉时间步 embedding。
上下采样模块使用 Unet 中的实现，支持池化和插值。（音频再考虑上采样用转置卷积）

至于 GAN 的损失先不考虑。

---

## Appendix: Checkerboard Artifacts

我们举一个简单的一维的例子，假设输入信号为 `[a, b]`，使用卷积核 `[w1, w2, w3]`，步长为 2 进行转置卷积。
第一步先膨胀，得到 `[a, 0, b]`，然后两边添加 `k-1-p` 个 0（假设 padding 为 p，kernel size 为 k），得到 `[0, 0, a, 0, b, 0, 0]`。
然后进行 stride=1 的卷积操作，得到输出信号 `[w3*a, w2*a, w1*a + w3*b, w2*b, w1*b]`。（实际操作过程中会适当加入padding或者删掉元素，使之满足刚好为原来的 2 倍长度，这里只是为了说明问题，就不展开了）

可以发现，输出信号的中间位置 `w1*a + w3*b` 受到了两个输入信号的影响，而其他位置只受到了一个输入信号的影响。这就导致了输出信号中间位置的值和其他位置的值在统计上是不一样的，从而产生了棋盘格效应。

如果使用最近邻插值，同样的例子，输入信号为 `[a, b]`，步长为 2 进行上采样。
第一步先插值，得到 `[a, a, b, b]`，这一步是均匀的。然后进行分辨率不变的卷积操作，kernel 为 `[w1, w2, w3]`，两边添加 `p` 个 0，得到 `[0, a, a, b, b, 0]`。
然后进行 stride=1 的卷积操作，得到输出信号 `[w2*a, w1*a + w2*a, w2*b + w3*b, w2*b]`。
可以发现，输出信号的每个位置都受到了相同数量的输入信号的影响，从而避免了棋盘格效应。


