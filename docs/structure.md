# JSA 项目结构与说明文档

本文档主要介绍 JSA (Joint Stochastic Approximation) 项目的整体架构、核心计算原理以及关键类的抽象接口作用。本项目基于 `pytorch_lightning` 构建，兼顾了代码的可复用性与灵活性。

核心框架分为两个主要的 Lightning 组件：
- **`LightningModule`**：管理模型结构、前向传播、采样过程、损失计算与优化器定义。本项目中该基类核心实现为 `JSA`。
- **`LightningDataModule`**：管理数据集加载、切分与预处理。

对应的实验配置保存在 `configs/` 文件夹中，支持通过 Hydra 等方式进行实例化。训练日志及结果保存在 `egs/` 的子目录下。

## 目录结构

```text
.
├── configs/                       # 配置文件夹
│   ├── latent_transformer/          # 条件 Transformer 配置文件
│   ├── jsa/                       # JSA 配置文件
│   └── vq_gan/                    # VQ-GAN 配置文件
├── data/                          # 数据集文件夹
│   ├── cifar10/                   # CIFAR10 数据集
│   └── mnist/                     # MNIST 数据集
├── docs/                          # 文档文件夹
│   ├── assests/                   # 资源文件
│   └── ...                        # 文档 Markdown 文件
├── egs/                           # 实验结果及日志记录存放文件夹
├── scripts/                       # 运行脚本
│   ├── infer.py                   # 推断/测试脚本
│   ├── monitor.py                 # 监控脚本
│   └── train.py                   # 训练脚本
├── src/                           # 核心代码
│   ├── base/                      # 基础模块（抽象类等）
│   ├── data/                      # 数据集相关实现
│   ├── models/                    # 模型抽象及顶层方法实现 (JSA, VQ-GAN等)
│   ├── modules/                   # 具体网络模块和组件
│   │   ├── gpts/                  # GPT 相关实现
│   │   ├── jsa/                   # JSA 详细组件
│   │   ├── losses/                # 损失相关实现
│   │   └── vqvae/                 # VQ-VAE 组件
│   ├── samplers/                  # 采样器实现
│   └── utils/                     # 工具类和回调函数
├── LICENSE                        # 许可证
├── README.md                      # 项目说明
├── requirements.txt               # 依赖库清单
├── run.sh                         # 运行脚本
└── test.py                        # 测试脚本
```

## JSA 原理框架

JSA 方法主要由两个子模型组成：生成模型 $p_\theta (x, h)$ 和辅助推断模型 $q_\phi (h|x)$。

我们的训练目标是同时最大化数据的对数似然，以及最小化生成模型与辅助推断模型后验分布之间的差异：
$$
\begin{cases}
\min KL(\tilde{p}(x)||p_\theta(x)) \\
\min \mathbb{E}_{\tilde{p}(x)} \big[KL(p_\theta(h|x)||q_\phi(h|x))\big]
\end{cases}
$$

通过推导，可以得到计算梯度的公式。设 $z = (\theta, \phi)$，采用以下联合形式来指导模型的梯度更新：
$$
F_\lambda(z) \triangleq \begin{pmatrix}
\sum_{i=1}^n \delta(\kappa = i) \nabla_\theta \log p_\theta(x_i, h_i) \\
\sum_{i=1}^n \delta(\kappa = i) \nabla_\phi \log q_\phi(h_i|x_i)
\end{pmatrix}.
$$
其中，$\kappa$ 是样本的索引变量，$h_i$ 则是从生成模型后验分布 $p_\theta(h|x_i)$ 中采样得到的隐变量样本。

## JSA 核心接口与模块设计

为了实现上述机制，对应的代码逻辑高度模块化，拆分为生成模型（Joint Model）、推断模型（Proposal Model）与采样器（Sampler）等组件。

### 1. `BaseJointModel` (生成模型)

代表分布 $p_\theta(x, h) = p_\theta(x|h) p(h)$，用于重构和联合概率计算。主要提供的接口：

- `log_joint_prob(x, h)`：计算联合概率分布的对数 $\log p_\theta(x, h)$。它由先验 $\log p(h)$ 和似然 $\log p_\theta(x|h)$ 两部分组合而成。
- `get_loss(x, h)`：用于模型优化过程，计算并返回关于给定隐状态和数据的损失（负对数似然或基于分布约定的损失如 MSE, BCE 等）。
- `sample(h, num_samples)`：从似然分布 $p_\theta(x|h)$ 中采样还原出数据 $x$。
- `decode(h)`：根据隐变量 $h$ 解码得到最可能的数据输出（如高斯分布的均值或二项分布的高概率输出）。

### 2. `BaseProposalModel` (辅助推断模型)

代表条件分布 $q_\phi(h|x)$，在采样阶段提供建议分布。主要接口：
- `log_conditional_prob(h, x)`：计算给定数据 $x$ 时，隐变量 $h$ 的条件对数概率 $\log q_\phi(h|x)$。
- `sample_latent(x)`：基于输入样本 $x$，从近似后验 $q_\phi(h|x)$ 中抽样得出隐变量 $h$ 的提议。
- `encode(x)`：确定性推断出给定数据下最可能的隐状态 $h$（例如概率大于0.5或最大概率对应的类）。
- `get_loss(h, x)`：用于推断模型的优化评估，返回负对数条件概率。

### 3. `MISampler` (Metropolis Independence Sampler)

专用于从目标概率（由 Joint Model 提供）中采样隐变量 $h \sim p_\theta(h|x)$。该类独立于主网络外接管采样过程，主要特性和接口：
- `step(x, idx, h_old)`：在每一步执行中，使用 Proposal Model 提议新的样本 $h'$，然后利用 Joint Model 和 Proposal Model 的概率密度计算接受率（MCMC机制），并进行接受或拒绝操作。
- **Cache 机制**：为加速采样和保证 MCMC 马尔可夫链连续性，针对数据集样本开启 Cache （缓存）。需要 Dataset 额外输出数据索引 `idx`。Sampler 记录每个样本当前的隐变量状态，并在后续 epoch 用作提议的初始值。
- `sample(...)`：多步并行采样的主要入口。

> 这里引入 cache 的一个劣势是，当数据集较大的时候，内存中需要维护一个与数据集大小相同的隐变量状态列表，可能会占用较多内存资源。

### 4. `JSA` (LightningModule 核心包装)

封装并协调以上所有流程，对接 PyTorch Lightning：
- 保存 `joint_model`, `proposal_model` 和 `sampler` 实例。
- `training_step`：获取数据后，先调用 `sampler` 获取经由 MCMC 处理的离散隐变量 $h$，再分别计算和收集 `joint_model` 及 `proposal_model` 对应的 loss 并执行梯度更新。
- 自动处理配置文件的组合实例化，例如通过 `hydra.utils.instantiate` 获取各模块实例。

### 5. `JsaDataset` 与数据加载

对 PyTorch `Dataset` 进行了一层抽象封装，为了支持缓存系统：
- 每一次调用 `__getitem__(index)`，除了返回原始的数据和标签内容外，**必须一并返回对应的全局数据索引 `index`**。这使得 Lightning 能够追踪批次数据在全局 Cache 中所对应的旧隐变量，维持正常的 MCMC 更新。

## JSA + Perceptual Loss

上面是最基本的 JSA 框架，后续需要引入 Perceptual Loss 来提升生成质量，因此我们需要在原始的 JSA 上面封装。

主要改变的模块有：

- `PerceptualJSA`：继承自 `JSA`，需要新加入三个阶段的训练过程，切换的逻辑。以及实现对能量函数的更新计算。
- `PerceptualJointModel`：原来的 `JointModel` 默认为高斯的似然计算。现在需要改为支持 Perceptual Loss 的计算，可以修改逻辑，设置不同的方法，计算不同的重建损失。

需要引入对 `h` 先验建模的 transformer 网络。

则 `JointModel` 里面的结构需要改变一下，需要两个网络，一个计算 distortion，一个计算 prior。然后在 `get_loss` 的时候根据当前的训练阶段，计算不同的损失。
> 原来的 `JointModel` 主要是高斯假设，分为似然加上均匀先验。现在我们无法得到似然，就需要修改设计，最后只要求得到联合的能量。


> 后续需要考虑，用不用额外定义一个计算损失的模块，还是就让 `JointModel` 来计算损失。
