# Code Summary

主要是对代码的一些总结和说明。
这里针对的主要是 [`Net2NetTransformer`](../taming/models/cond_transformer.py) 相关的代码，这个类针对的是 VQ-GAN 训练的第二个阶段，即训练一个 Transformer 来对 VQ-GAN 的离散潜在空间进行建模。

## `Net2NetTransformer`

输入的参数有：
- `transformer_config`: Transformer 的配置参数，通常是一个字典，包含了 Transformer 的各种超参数设置。
- `first_stage_config`: 第一阶段模型（通常是 VQ-GAN）的配置参数。
- `cond_stage_config`: 条件阶段模型的配置参数。
- `permuter_config`: 用于数据排列的配置参数。
- `ckpt_path`: 预训练模型的路径，如果有的话，可以加载预训练的权重。
- `ignore_keys`: 在加载预训练模型时忽略的键。
- `first_stage_key`: 第一阶段模型返回数据集的键
- `cond_stage_key`: 条件阶段模型返回数据集的键
- `downsample_cond_size`: 条件大小的下采样因子。
- `pkeep`: 保持概率，用于 Dropout。
- `sos_token`: `int`，起始标记的索引。
- `unconditional`: `bool`，是否为无条件模型。

### `first_stage_config`

这里用于加载 VQ-GAN 模型的配置，通常包含编码器和解码器的参数。还设置了 `ckpt_path`，用于加载预训练的 VQ-GAN 权重。

需要注意的是，这里的 `loss_config` 被设置为一个 `DummyLoss`，表示在这个阶段不计算损失，因为 VQ-GAN 已经在第一阶段训练完成。
> dummy 意思是“虚拟的”或“占位的”，表示这里的损失函数只是一个占位符，不会实际计算任何损失。

```python
class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()
```

第二点需要注意的就是，为了避免 `lightning` 在训练时可能自动调用父类的 `train()` 方法，使得这里的 VQ-GAN 模型进入训练模式，从而影响其行为，这里并没有只设置 `model.eval()`，而是使用了 python 的 monkey patching 技术，通过将 `train` 方法替换为一个空函数，来确保模型始终处于评估模式。

```python
def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self
```
> 这里返回 `self` 是为了遵守 `pytorch` 的返回 API 规范，`train()` 方法通常返回模型本身，以便支持链式调用，例如 `model.train().cuda()`。

后面进行赋值：

```python
model.train = disabled_train
```

> monkey patching 是 python 中的一种动态修改类或模块行为的技术，可以在运行时改变类或模块的方法和属性。其不需要修改原始代码，而是通过定义一个新的函数或方法，然后将其赋值给类或模块的相应属性，从而实现对其行为的修改。优点是灵活且方便，缺点是可能导致代码难以理解和维护，因为修改后的行为可能与原始代码不一致。

### `cond_stage_config`

用于加载条件模型。将图像生成视为一个序列预测任务，而这里的条件模型的作用，就是接受条件信息（另一张图片、语义图或者被遮挡的图片），通过 `encode` 方法将其编码为离散的表示，作为 Transformer 的输入 `prompt`。

代码中，允许有三种可能的选择：

1. 复用第一阶段模型（`__is_first_stage__`），此时直接使用 VQ-GAN 的编码器来编码条件图像。
2. 无条件/类别条件（`__is_unconditional__` or `SOSProvider`），此时为标准的随机生成，或者根据类别标签生成。此时条件模型变为一个简单的 `SOSProvider`（Start of Sequence Provider），只提供一个起始标记。
```python
class SOSProvider(AbstractEncoder):
    # for unconditional training
    def __init__(self, sos_token, quantize_interface=True):
        super().__init__()
        self.sos_token = sos_token
        self.quantize_interface = quantize_interface

    def encode(self, x):
        # get batch size from data and replicate sos_token
        c = torch.ones(x.shape[0], 1)*self.sos_token
        c = c.long().to(x.device)
        if self.quantize_interface:
            return c, None, [None, None, c]
        return c
```
3. 独立预训练的条件模型，根据 `cond_stage_config` 中的配置加载相应的模型。这里的模型同样是参数冻结的。

这里的一个独立的模型的例子：
```python
class CoordStage(object):
    def __init__(self, n_embed, down_factor):
        self.n_embed = n_embed
        self.down_factor = down_factor

    def eval(self):
        return self

    def encode(self, c):
        """fake vqmodel interface"""
        assert 0.0 <= c.min() and c.max() <= 1.0
        b, ch, h, w = c.shape
        assert ch == 1

        c = torch.nn.functional.interpolate(
            c, scale_factor=1 / self.down_factor, mode="area"
        )
        c = c.clamp(0.0, 1.0)
        c = self.n_embed * c
        c_quant = c.round()
        c_ind = c_quant.to(dtype=torch.long)

        info = None, None, c_ind
        return c_quant, None, info

    def decode(self, c):
        c = c / self.n_embed
        c = torch.nn.functional.interpolate(
            c, scale_factor=self.down_factor, mode="nearest"
        )
        return c
```
上面的 `CoordStage` 类实现了一个简单的条件编码器，先对原始图像进行下采样，然后将像素值量化为 $[0, n\_embed]$ 范围内的离散值，最后返回这些离散值的索引作为编码结果。解码过程则是将离散值还原为像素值，并进行上采样。


### `first_stage_key` 和 `cond_stage_key`

这两个针对的是数据集中返回的键名，用于从数据集中提取相应的数据。`first_stage_key` 通常是图像数据，而 `cond_stage_key` 则是条件数据（例如另一张图像、语义图等）。

一个简单的例子，假设 `first_stage` 返回的数据集是下面的：
```json
{
    "image": [B, C, H, W],
    "class_label": [B],
}
```
则我们可以设置 `first_stage_key` 为 `"image"`，提取出图像数据。

而 `cond_stage_key` 有多种可能，常见的有：

- `"image"`：表示条件是另一张图像。
- `"depth"`：表示条件是深度图。
- `"segmentation"`：表示条件是语义分割图。
- `"class_label"`：表示条件是类别标签。


### `pkeep`

是 Probability of Keeping 的缩写，作用是在训练过程中，将部分图像 `token` 替换为随机噪声，从而增强模型的鲁棒性。其值通常在 0 到 1 之间，表示保留原始 `token` 的概率。

```python
if self.training and self.pkeep < 1.0:
    mask = torch.bernoulli(
        self.pkeep * torch.ones(z_indices.shape, device=z_indices.device)
    )
    mask = mask.round().to(dtype=torch.int64)
    r_indices = torch.randint_like(
        z_indices, self.transformer.config.vocab_size
    )
    a_indices = mask * z_indices + (1 - mask) * r_indices
else:
    a_indices = z_indices
```

让模型学会纠错。

> 需要注意的是，这并不是训练 transformer 的标准做法，因为对于文本而言，信息密度是很高的，变换一个 token 可能意思大变。但对于图像而言，信息是高度冗余的，部分区域的变化并不会对整体图像造成太大影响。因此，这种做法在图像生成任务中是可行且有效的。

### `permuter_config`

Transformer 通常是对一维序列进行建模，而图像是二维的，因此需要将二维的图像 `token` 展开为一维序列。`permuter` 的作用就是定义这种展开的方式。

这里定义了一系列的 [`Permuter`](../taming/modules/transformer/permuter.py) 类，但是代码中唯一使用的就是 `Identity` 类，表示不进行任何变换，直接按行优先的顺序展开。

```python
class Identity(AbstractPermuter):
    def __init__(self):
        super().__init__()

    def forward(self, x, reverse=False):
        return x
```

下面简单介绍一下剩余的几种 `Permuter`：

- `Subsample`：迭代地将图像分为 $2 \times 2$ 的块，然后每一层级将 “粗粒度” 的放在前面。试图构造一种多尺度的结构。
- `ZCurve`：使用 Z 曲线（Morton order）对图像进行遍历，这是一种空间填充曲线，可以保持空间局部性。
- `SpiralOut`：从图像的中心开始，沿着螺旋路径向外遍历图像。
- `SpiralIn`：与 `SpiralOut` 相反，从图像的边缘开始，沿着螺旋路径向内遍历图像。
- `Random`：随机打乱图像的遍历顺序。
- `AlternateParsing`：蛇形遍历图像，即第一行从左到右，第二行从右到左，依此类推。

这里展示两个典型的顺序：

| Permuter | 顺序示意图 |
|----------|-------------|
| ZCurve   | ![ZCurve](./assets/permuter_zcurve.png) |
| SpiralOut| ![SpiralOut](./assets/permuter_spiralout.png) |

> `Permuter` 针对的也只有图像 `token` 部分，条件 `token` 部分保持不变。

### `forward()`

前向传播的逻辑就是，输入两个部分：图像 `x` 和条件 `c`，通过 `first_stage` 和 `cond_stage` 分别编码为离散的 `token`，然后将条件 `token` 作为 `prompt`，拼接到图像 `token` 的前面，一起输入到 Transformer 中进行训练。

其中注意的几点：

1. 上面说的随机替换 `token` 的逻辑，只在图像 `token` 上进行，条件 `token` 保持不变。
2. 拼接 `prompt` 时，条件 `token` 在前，图像 `token` 在后。
3. 计算损失时，只计算图像 `token` 部分的交叉熵损失，条件 `token` 部分不计算损失。

> 我们看到，由于图像的序列的长度是完全固定的，因此这里不需要加入 `<eos>` 标记来表示序列结束。

#### `encode_to_z_indices()`

调用 `first_stage.encode()` 方法，将图像编码为离散的 `token` 索引。

#### `encode_to_c_indices()`

调用 `cond_stage.encode()` 方法，将条件图像编码为离散的 `token` 索引。

这里设置了一个 `downsample_cond_size` 参数，用于对条件 `c` 进行下采样到固定的大小，然后编码为 `token`。


### `sample()` 方法

这是 Transformer 的核心生成逻辑，自回归地生成图像 `token` 序列。

原来的代码有一些问题，这里先不对其进行修改。

需要的参数有：

- `x`: 初始的图像 `token` 序列（或许设置为 `z` 更加统一），可以设置为长度为 0 的张量，表示从头开始生成；也可以设置为部分生成的序列，表示从该序列继续生成。- `c`: 条件 `token` 序列，作为 `prompt`。
- `steps`: 需要生成的图像 `token` 的数量。
- `temperature`: 采样的温度参数，控制生成的多样性。
- `top_k`: 采样时只考虑概率最高的前 `k` 个选项。
- `callback`: 每生成一个 `token` 后调用的回调函数。

需要注意的是，这里的 `x` 的长度，和 `steps` 加起来应该等于最终需要生成的图像 `token` 的总长度。如果超出来的话，输出的图像 `token` 序列的长度会超过预期。
> 一个例子，我们一张图片被编码为 256 个 `token`，如果我们传入的 `x` 长度为 100，`steps` 为 200，那么最终生成的图像 `token` 序列长度就是 300，而不是 256。
> 代码中并未做截断处理。

正常的逻辑：
```python
x = torch.cat((c, x), dim=1)
block_size = self.transformer.get_block_size() # Max context size
assert not self.transformer.training
for k in range(steps):
    callback(k)
    assert x.size(1) <= block_size  # make sure model can see conditioning
    x_cond = (
        x if x.size(1) <= block_size else x[:, -block_size:]
    )  # crop context if needed
    logits, _ = self.transformer(x_cond) # [B, T, vocab_size]
    # pluck the logits at the final step and scale by temperature
    logits = logits[:, -1, :] / temperature
    # optionally crop probabilities to only the top k options
    if top_k is not None:
        logits = self.top_k_logits(logits, top_k)
    # apply softmax to convert to probabilities
    probs = F.softmax(logits, dim=-1)
    # sample from the distribution or take the most likely
    if sample:
        ix = torch.multinomial(probs, num_samples=1)
    else:
        _, ix = torch.topk(probs, k=1, dim=-1)
    # append to the sequence and continue
    x = torch.cat((x, ix), dim=1)
    # cut off conditioning
    x = x[:, c.shape[1] :]
return x
```

串行采样，每次生成一个 `token`，然后将其拼接到序列的末尾，继续生成下一个 `token`，直到生成指定数量的 `token` 为止。

但是文中还增加了另一个判断，是并行一次性地生成所有 `steps` 个 `token`，
```python
 if self.pkeep <= 0.0:
    # one pass suffices since input is pure noise anyway
    assert len(x.shape) == 2
    noise_shape = (x.shape[0], steps - 1)
    # noise = torch.randint(self.transformer.config.vocab_size, noise_shape).to(x)
    noise = c.clone()[:, x.shape[1] - c.shape[1] : -1]
    x = torch.cat((x, noise), dim=1)
    logits, _ = self.transformer(x)
    # take all logits for now and scale by temp
    logits = logits / temperature
    # optionally crop probabilities to only the top k options
    if top_k is not None:
        logits = self.top_k_logits(logits, top_k)
    # apply softmax to convert to probabilities
    probs = F.softmax(logits, dim=-1)
    # sample from the distribution or take the most likely
    if sample:
        shape = probs.shape
        probs = probs.reshape(shape[0] * shape[1], shape[2])
        ix = torch.multinomial(probs, num_samples=1)
        probs = probs.reshape(shape[0], shape[1], shape[2])
        ix = ix.reshape(shape[0], shape[1])
    else:
        _, ix = torch.topk(probs, k=1, dim=-1)
    # cut off conditioning
    x = ix[:, c.shape[1] - 1 :]

```

这种方式的前提是，`pkeep` 设置为 0.0，表示完全不保留原始的图像 `token`，而是全部使用随机噪声进行替换。这样的话，模型根本不需要参考之前生成的 `token`，因为它们都是噪声，因此可以一次性地生成所有 `steps` 个 `token`。
实际生成方式中，应该不会使用这种方式，因为这样生成的图像质量会比较差，缺乏连贯性。

#### `top_k_logits()` 方法

用于在采样时，只保留概率最高的前 `k` 个选项，将其他选项的概率设置为负无穷大，从而在采样时不会选择这些选项。

### `log_images()` 方法

主要展示下面的 5 类结果：

1. 原始输入：
  - `log["inputs"]`: 原始图像 `x`。
  - `log["conditions"]`: 条件图像 `c`。
2. 重建结果：
  - `log["reconstructions"]`: 通过 `first_stage` 对原始图像进行编码和解码得到的重建图像。
3. 条件生成参考：
  - `log["conditioning_rec"]`: 对条件信息 `c` 编码又解码得到的可视化结果。
4. 随机生成结果：
  - `log["samples_half"]`: 使用条件 `c`，给模型一半真实的图像 `token`，生成剩余一半的图像 `token`，然后解码得到的图像。
  - `log["samples_nopix"]`: 使用条件 `c`，完全不提供任何真实的图像 `token`，全部使用随机噪声生成图像 `token`，然后解码得到的图像。
5. 确定性生成结果：
  - `log["samples_det"]`：使用条件 `c`，不提供真实的图像 `token`，但是采用 Greedy 采样（即温度为 0），生成图像 `token`，然后解码得到的图像。

### `configure_optimizers()` 方法

这里对不同的参数进行差异性的权重衰减（weight decay）策略，而这种策略源自 minGPT，为了提高 Transformer 的训练稳定性。

需要衰减的：
- 线性层权重

不需要衰减的：
- 线性层偏置
- LayerNorm 和 Embedding 层的所有参数
- 位置编码

## `GPT`

这是实际的 Transformer 实现类，继承自 `nn.Module`。其来源于 [minGPT](https://github.com/karpathy/minGPT)。

主要包含以下几个部分：

- `CausalSelfAttention`：实现了自注意力机制，包含多头注意力和投影层。
- `Block`：Transformer 的基本模块，包含一个自注意力层和一个前馈神经网络层。
- `GPT`：整体的 Transformer 模型，包含多个 `Block` 叠加而成。

基本上是标准的 GPT-2 的实现。需要注意的几点：

1. 位置编码：使用了可学习的位置编码，而不是固定的位置编码。
```python
self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
```
2. 权重共享：确认了一下，标准的 GPT-2 模型是有权重共享的，就是 `_tied_weights_keys = {"lm_head.weight": "transformer.wte.weight"}`，表示输出层的权重和输入嵌入层的权重是共享的。但是在 `minGPT` 的实现中，并没有这一个约束，默认是不共享的。这里是修改的 `minGPT` 的代码，也没有加入权重共享的逻辑。

这里还定义了 `sample()` 和 `sample_with_past()` 方法，用于自回归地生成序列，和上面的 `Net2NetTransformer.sample()` 方法类似。（可以说后者就是复写了这里的 `sample` 的逻辑，没有设置 KV cache）

> 代码中还有一个 `CodeGPT` 类，有点问题，而且也没有用到，这里就不再介绍。

## `sos_token`

这是我在训练中遇到的一个问题，就是我们在做无条件生成的时候，模型需要一个起始标记 `sos_token` 来表示序列的开始。
而我们 VQ-GAN 的词汇表大小是 1024，其每一个 token 都是一个有效的图像 `token`，因此我们需要为 `sos_token` 选择一个不冲突的索引。
因此我选择了 1024 作为 `sos_token` 的索引，这样就不会和任何图像 `token` 冲突。因此，transformer 的词汇表大小需要设置为 1025。

但这样出现的问题就是，我们的 transformer 在训练时，可能会预测出 `sos_token` 这个索引，这样解码时，decoder 会报错，因为没有对应的图像 `token`。

我去看了一下 `GPT-2` 怎么处理这个问题的，首先，`GPT-2` 的词汇表中有一些特殊的 token，例如 `<|endoftext|>`，这些 token 本身就是合法的 `token`，并不会报错。而且 `GPT-2` 在生成时，即使是无条件生成，也必须有一个起始标记 `<|endoftext|>`，这样模型才能知道序列的开始。

> 与我们现在的问题不对应

解决的思路，一种是，在输出 logits 后，将 `sos_token` 的 logits 设置为负无穷。但是需要修改的地方会很多。
另一种是，我的词表就限制在 0-1023 之间，不允许预测出 1024 这个索引。起始的输入我们设置一个可学习的嵌入向量。
> 我们可以认为，出现这种情况的几率很小，直接判断，如果预测的出现了 `sos_token`，就设置为特定的索引，例如 0，或者直接随机一个合法的索引，这样解码时就不会报错了。


## 需要修改的部分

首先，我们需要先跑一个 VQ-GAN 的第一阶段训练，得到一个预训练的 VQ-GAN 模型，然后才能进行第二阶段的 Transformer 训练。
然后尝试一下原来的代码是否能运行。
最后看如何往现在的 JSA 框架中移植这些代码。


