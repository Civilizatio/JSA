# Cond Transformer Net Exploration

这里主要针对 VQ-GAN 的第二个阶段的训练修改代码。
VQ-GAN 本身的代码看了一下，需要重新组织一下，适配我们现在的训练框架。

需要实现的子模块部分，包括：
原来的训练模块叫做 `Net2NetTransformer`，现在改名为 `LatentTransformer`，主要包含以下几个模块：

- `first_stage_model`：VQ-GAN 的第一个阶段的训练模型。JSA or VQ-VAE。
    - 设置参数为 `config_path`, `ckpt_path`。
- `transformer_net`：条件变换器网络，VQ-GAN 的第二个阶段的训练模型。
- `cond_stage_model`：条件阶段模型，一个额外的第三方的预训练模型，也可能为 VQ-GAN 的第一个阶段的训练模型。
    - 设置为类似的 `class_path` 和 `init_args` 的参数。相当于一个预训练模型的加载。这里考虑的情形有三种：
    1. 直接使用 VQ-GAN 的第一个阶段的训练模型作为条件阶段模型。
    2. 使用一个预训练的第三方模型作为条件阶段模型。
    3. 不使用条件阶段模型。（无条件生成，需要加入参数 `sos_token`）
    > 这个实现如何做到呢，如果为第一种情形，直接设置一个配置为 `__first_stage_model__`，在加载的时候进行特殊处理即可。第二种情形，我们需要给定第三方模型的 `class_path` 和 `init_args`，在加载的时候进行实例化。以及需要指定 `ckpt` 的路径。第三种情形，我们是可以与第二种情形合并的，设置一个 `SOSProvider` 的类路径，在加载的时候实例化即可。
    > 则后两种是能统一为 `class_path` 和 `init_args` 的形式的。但是第一种我们上面给定的是 `first_stage_model_config` 的路径以及 `first_stage_model_ckpt` 的路径，不是 `class_path` 和 `init_args` 的形式。我想统一一下，如何统一？
    > 不打算设置为统一的 `class_path` 和 `init_args` 的形式了，新增加一个 `mode` 参数，来区分三种情形：
    > - `mode` = `first_stage_model`：使用 VQ-GAN 的第一个阶段的训练模型作为条件阶段模型。
    > - `mode` = `external_model`：使用一个预训练的第三方模型作为条件阶段模型。
    > - `mode` = `no_cond`：不使用条件阶段模型。（无条件生成，需要加入参数 `sos_token`）
    > 然后后面加入不同的参数，如果为 `first_stage_model`，则不需要额外参数；如果为 `external_model`，则需要 `class_path`、`init_args` 和 `ckpt`；如果为 `no_cond`，则需要 `sos_token`。

- `permuter`：按照一定的顺序，将二维的图像块进行排列，形成一维的序列输入到变换器网络中。这个已经有实现的类了，可以直接使用。

需要的参数比较重要的就是：
- `pkeep`：在训练过程中，按照一定的概率，保持原始的输入不变，或者进行随机的替换。这个直接采用。

所以一个样例就是：
```yaml
model:
    class_path: src.models.latent_transformer.LatentTransformer
    init_args:
        first_stage_config:
        # if config_path is present, then load the model from the config and ckpt path. Otherwise, load the model from the class_path and init_args.
            config_path: ./configs/first_stage_model.yaml
            ckpt_path: ./egs/first_stage_model/version_0/checkpoints/best-checkpoint.ckpt

            class_path: src.models.vq_gan.VQGAN
            init_args:
                image_size: 256
                in_channels: 3
                # others...
        cond_stage_config: 
            mode: first_stage_model # or external_model or no_cond
            # if mode is external_model, then need to specify the class_path, init_args and ckpt_path
            class_path: src.models.some_model.SomeModel
            init_args:
                # some args for the external model
            ckpt_path: ./egs/some_model/version_0/checkpoints/best-checkpoint
            # if mode is no_cond, then need to specify the sos_token
            # class_path: src.models.sos_provider.SOSProvider
            # init_args:
            #     sos_token: 0

            # if mode is first_stage_model, then no need to specify the above parameters
        transformer:
            class_path: src.modules.gpts.mingpt.GPT
            init_args:
                # some args for the transformer net
        permuter:
            class_path: src.modules.permuter.SomePermuter
            init_args:
                # some args for the permuter
        pkeep: 0.9
        cond_stage_dataset_key: some_dataset_key # cond_stage_model 需要用到的数据集的 key，可能与 first_stage_model 的数据集的 key 不同，因此需要单独列出来。


```
相应的代码：

```python
from lightning.pytorch import LightningModule
from xxx import AbstractPermuter, AbstractFirstStageModel, AbstractCondStageModel, AbstractTransformerNet
from xxx import initialize_from_config

class LatentTransformer(LightningModule):
    def __init__(self,
        first_stage_config: dict,
        cond_stage_config: dict,
        transformer: GPT,
        permuter: AbstractPermuter,
        pkeep=0.9,
        cond_stage_dataset_key="image",
    ):
        super().__init__()
        self.first_stage_model = initialize_from_config(first_stage_config)
        self.cond_stage_model = initialize_from_config(cond_stage_config)
        self.transformer_net = transformer
        self.permuter = permuter
        self.pkeep = pkeep
        self.cond_stage_dataset_key = cond_stage_dataset_key


```



