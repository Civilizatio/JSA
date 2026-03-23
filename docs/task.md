# JSA + Perceptual Loss

基本的 JSA 原理，可以在 [theory.md](./theory_of_jsa.md) 中查看。我们目前的代码框架和接口设计，可以在 [structure.md](./structure.md) 中查看，基本上是按照 JSA 的原理框架来设计的。此外，还额外从 VQ-VAE 中假如了 VQ-VAE 的实现以及 Latent-transformer 的实现。

后续我们的任务是在现有的 JSA 高斯似然的基础上，引入 Perceptual Loss 来提升生成质量。引入的 perceptual loss 的原理，可以在 [jsa_perceptual_loss.md](./jsa_perceptual_loss.md) 中查看。主要就是能量的建模，以及为了训练，引入了三个阶段的训练步骤。

下面主要介绍可能的代码实现的细节。

## Code Implementation Details

### 三个阶段如何实现

继承当前的 `JSA` 类，定义一个新的 `PerceptualJSA` 类。其功能主要是接受新的参数，来控制调整训练的三个阶段。

需要注意的是，我需要的是三个阶段分开训练。也就是说，先训练第一阶段。然后经过我观察，确定可以后，再重新跑实验训练第二阶段。然后再训练第三阶段。也就是说，三个阶段的训练是分开进行的，而不是在一个训练过程中自动切换三个阶段。这样的好处就是，我可以跑很多个第一阶段得到的模型，然后再选择一个最好的模型来继续训练第二阶段。然后再选择一个最好的模型来训练第三阶段。

当然，一个设计就是，引入一个 `force_stage` 的参数。来控制当前训练的阶段。同时还有默认的自动切换阶段的设计。也就是说，如果 `force_stage` 不为 None，则强制使用这个阶段来训练。否则的话，就根据当前的 epoch 来自动切换阶段。

我希望我的配置文件 `yaml` 也是三个阶段分开的。我现在已经在 [configs/perceptual_jsa/](../configs/perceptual_jsa/) 下列出了文件的示例，一个 `jsa_base.yaml` 是公共的配置文件，包含了三个阶段的公共配置。然后每个阶段有一个单独的配置文件，来覆盖不同阶段的不同配置。

### JointModel 的设计

我们一开始的 `JointModel` 的设计，是基于高斯似然的设计。也就是说，我们的重建损失是基于高斯似然来计算的，先验使用均匀先验，比较容易获得。但是现在我们引入了 Perceptual Loss，我们无法直接计算似然了。也就是说，我们无法直接计算 $p_\theta(x|h)$ 的似然了。因此，我们需要修改 `JointModel` 的设计，来适应新的损失计算。

其基本的逻辑不变，就是需要返回的是一个能量值。这个能量值包含了 distortion 和 prior 两部分。distortion 是基于 pixel loss 和 perceptual loss 来计算的，prior 是基于先验来计算的。然后在 `get_loss` 的时候，根据当前的训练阶段，来计算不同的损失。

则根据上面的设置，可以将修改前后的统一起来，我们的 `JointModel` 不再是
```python
class JointModelCategoricalGaussian(BaseJointModel):
    """
    p_theta(x, h), must implement:
        - log_joint_prob(x, h)
    We assume Categorical prior for p(h) and Gaussian likelihood for p(x|h).
    """

    def __init__(
        self,
        net: nn.Module,
        sigma=0.1,
        sigma_mode: str = "learnable",  # whether sigma is learnable or fixed or scheduled
        sample_chunk_size=8,
    ):
        pass
```
而是修改为
```python
class JointModelCategoricalEnergy(BaseJointModel):
    """
    p_theta(x, h), must implement:
        - log_joint_prob(x, h)
    We assume Categorical prior for p(h) and Energy-based likelihood for p(x|h).
    """

    def __init__(
        self,
        distortion_model: nn.Module,  # the model to calculate distortion, e.g. pixel loss or perceptual loss or both
        prior_model: nn.Module,  # the model to calculate prior, e.g. uniform prior or learned prior
        distortion_weight=1.0,
        prior_weight=1.0,
        sample_chunk_size=8,

    ):
        pass
```

这样我们可以在原来的似然的基础上扩展，计算 distortion 的部分，在原来的先验的基础上扩展，计算 prior 的部分。我们可以引入 [gpt](../src/modules/gpts/mingpt.py) 来建模先验，来计算 prior 的能量值。

原来的 `sigma` 是一个可学习或者可调整值，这里可能没有了，或者者说它被 absorbed 进了 distortion_weight 里面了。因为我们不再直接计算似然了，而是计算一个能量值，这个能量值包含了 distortion 和 prior 两部分。但是会引入一个新的可调参数，为 $\lambda$，来控制 prior 的权重，因为我们的第二阶段是逐步增加 prior 的权重到 1 来训练的。

### MISampler 的调整

有可能随着 `JointModel` 的修改，我们的 `MISampler` 用到的接口也需要调整。

### Langevin Dynamics 的实现

由于第三阶段需要引入对 $x$ 的 LD 采样。这里需要在 [samplers/](../src/samplers/) 下实现一个新的 `LangevinSampler`，来实现 LD 采样的功能。这个 `LangevinSampler` 需要接受一个 `JointModel`，来计算能量值的梯度，然后根据 LD 的更新公式来更新 $x$ 的值。
