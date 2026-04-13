# JSA + Perceptual Loss + NCG

本次代码修改的目标是：

1. 完善 [NCGSampler](../src/samplers/ncg_sampler.py) 的实现，目前的实现有点问题，修改使之适配新的代码结构。
2. 修改 [PerceptualJSA](../src/losses/perceptual_jsa.py) 的实现，修改训练流程，使之在第一三阶段引入 NCG 采样器。

原理部分主要参考自：

- [jsa_perceptual.md](./jsa_perceptual_loss.md)
- [ncg_in_jsa_perceptual.md](./ncg_in_jsa_perceptual.md)

## 1. NCGSampler 的修改

当前的 NCG 的实现，存在以下问题：

- 只是根据 [joint model](../src/models/joint_model.py) 的 distortion 部分来进行采样，这在第一阶段是可以的，但是在第三阶段，我们需要的是对联合的能量进行采样，而不是单纯的 distortion。我的修改想法是，在 NCGSampler 中引入一个参数，来区分当前是第一阶段还是第三阶段，根据不同的阶段来选择不同的能量进行采样。或者不需要区分阶段，因为我的 joint model 会自动计算能量，不包含 prior 部分的能量的时候，采样的能量就是 distortion 的能量。（这个你看一下哪一个更加合理）。这里参考 [ncg_in_jsa_perceptual.md](../docs/ncg_in_jsa_perceptual.md) 中的描述，采用思路 2 实现。
- 对于 prior 的关于 embedding 的部分，当前的代码是没有的，我的想法是，在原有的 [GPT](../src/modules/gpts/mingpt.py) 模型中，增加一个参数，用于控制是否使用 weight tying，如果使用 weight tying，则 input embedding 和 output embedding 是同一个，如果不使用，则需要分别定义两个 embedding。然后在计算 prior 的能量的时候，根据是否使用 weight tying 来计算对应的梯度。而这里求梯度的逻辑最好不要放在 gpt 的实现中，而是放在 NCGSampler 中，因为这个是 NCG 特有的逻辑，不应该污染到 GPT 模型的实现中。而且 NCGSampler 能够接触到的是 [PriorModel](../src/modules/jsa/prior_model.py) 的接口，又封装了一层。这个你看一下怎么实现比较合理。

## 2. PerceptualJSA 的修改

我希望修改的主要有下面几点：

- 目前在第二阶段的训练中，我一开始是使用了 MIS 采样的 h，后面又采用了直接使用 encoder 直接编码，不经过采样的 h。我后面有可能对这两种方法进行对比实验。因此希望你能给我加一个参数，这两种方法可以切换的。

## 3. LD 采样的修改

主要针对的是 [Langevin Dynamics](../src/samplers/langevin_sampler.py) 采样器，目前没有接受拒绝的步骤，希望你能加上这个步骤，使其成为一个完整的 MALA 采样器。

## 4. JointNegativeSampler 的修改

主要针对的是 [JointNegativeSampler](../src/samplers/joint_negative_sampler.py) 采样器，目前的实现有点简陋，希望你重构一下。逻辑方面应该没有太大问题。

## 5. 注意事项

- 在修改代码的过程中，尽量保持代码的清晰和可读性，注释要充分，尤其是对于一些复杂的逻辑部分。
- 修改完成后，如果可以的话，进行简单地测试，确保修改的代码能够正常运行，并且没有引入新的 bug。
- ！！！ 修改完成后，给我一个 change.md 的文件，详细地说明你修改了哪些文件，修改了哪些函数，修改的逻辑是什么，以及为什么要这样修改。这个 change.md 的文件对于我理解你的修改非常重要，所以请务必写得详细一些。
