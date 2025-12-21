# JSA Structure

这里主要总结一下 JSA 的整体结构，必须的抽象的接口等。

后面打算使用 `pytorch_lightning` 来重构代码，更加方便的同时，增强可复用性。
`pytorch_lightning` 需要的两个主要的类是 `LightningModule` 和 `LightningDataModule`。
其中，

- `LightningModule` 主要负责模型的定义，前向传播，损失计算，优化器定义等。
- `LightningDataModule` 主要负责数据的加载，划分等。

目前的想法就是，不同的方法定义为不同的 `LightningModule`，现在只定义 `JSA` 一个类。
不同的数据集定义为不同的 `LightningDataModule`，例如二维高斯数据集，MNIST数据集等。

上面的都放在不同的文件夹中，例如 `models`，`data` 等。

实验的结果保存在 `egs` 文件夹中，每个数据集一个子文件夹，例如 `binary_mnist`，`continuous_mnist` 等。子目录下再根据不同的实验配置，保存不同的版本。

## 目录结构

```text
.
├── configs/                       # 配置文件夹
│   ├── bernoulli_prior_binary_mnist.yaml           # Bernoulli MNIST 实验配置文件
│   ├── categorical_prior_continuous_mnist.yaml     # Categorical MNIST 实验配置文件
│   └── ...                        
├── data/                          # 数据相关存储文件夹
│   ├── gaussian_2d/               # 二维高斯数据集
│   ├── mnist/                     # MNIST 数据集
│   └── ...                        
├── src/                           # 方法相关
│   ├── base/                      # 核心模块
│   │   ├── jsa_module.py          # 关于 proposal model 和 joint model 的抽象类定义
│   │   ├── base_sampler.py        # 抽象采样器定义
│   │   └── ...                    
│   ├── data/                      # 数据集定义
│   │   ├── gaussian_2d.py         # 二维高斯数据集
│   │   ├── mnist.py               # MNIST 数据集
│   │   └── ...                    
│   ├── models/                    # 模型定义
│   │   ├── jsa.py                 # JSA 方法实现
│   │   ├── components/            # 子模块定义
│   │   │   ├── proposal_model.py  # proposal 模型定义
│   │   │   ├── joint_model.py     # joint 模型定义
│   │   │   └── ...                
│   │   └── ...                    
│   ├── samplers/                  # 采样器定义
│   │   ├── misampler.py           # MIS 采样器实现
│   │   └── ...                    
│   └── utils/                     # 工具函数
├── scripts/                       # 运行脚本
│   ├── train.py     # 运行 Bernoulli MNIST 实验脚本
│   ├── run_categorical_MNIST.py   # 运行 Categorical MNIST 实验脚本
│   └── ...                        
├── docs/                          # 文档文件夹
│   ├── structure.md               # 结构说明
│   └── ...                        
├── egs/                           # 实验结果
│   ├── binary_mnist/              # Binary MNIST 实验
│   ├── continuous_mnist/          # Continuous MNIST 实验
│   └── ...                        
├── README.md                      # 项目说明
└── ...
```


## JSA 方法

### 原理

包括两个模型，生成模型 $p_\theta (x, h)$ 和辅助推断模型 $q_\phi (h|x)$。

我们的训练目标是最大化以下目标函数：
$$
\begin{cases}
\min KL(\tilde{p}(x)||p_\theta(x)) \\
\min \mathbb{E}_{\tilde{p}(x)} \big[KL(p_\theta(h|x)||q_\phi(h|x))\big]
\end{cases}
$$

也就是既要最大化数据的似然，又要最小化生成模型和辅助推断模型之间的差异。

因此写出梯度
$$
\begin{cases} g_\theta \triangleq -\nabla_\theta KL\left[\tilde{p}(x)||p_\theta(x)\right]  = \mathbb{E}_{\tilde{p}(x)p_\theta(h|x)} \left[ \nabla_\theta \log p_\theta(x, h) \right] \\ g_\phi \triangleq -\nabla_\phi \mathbb{E}_{\tilde{p}(x)} KL\left[p_\theta(h|x)||q_\phi(h|x)\right]  = \mathbb{E}_{\tilde{p}(x)p_\theta(h|x)} \left[ \nabla_\phi \log q_\phi(h|x) \right] \end{cases}
$$

> 这里用负号是因为我们使用 SA 的理论来求解，实际实现过程中，方向是相反的。

最后推导得到的计算梯度的公式为：

$$
F_\lambda(z) \triangleq \begin{pmatrix}
\sum_{i=1}^n \delta(\kappa = i) \nabla_\theta \log p_\theta(x_i, h_i) \\
\sum_{i=1}^n \delta(\kappa = i) \nabla_\phi \log q_\phi(h_i|x_i)
\end{pmatrix}.
$$

其中，这里的 $z = (\theta, \phi)$，$\kappa$ 是一个索引变量，表示从 $n$ 个样本中选择的样本的索引，$h_i$ 是从生成模型 $p_\theta(h|x_i)$ 中采样得到的隐变量。

### 代码实现

因此，实现的时候，需要两个模型，一个生成模型 `joint_model`，一个辅助推断模型 `proposal_model`。
- 生成模型 `joint_model` 需要提供的接口是：
  - `log_joint_prob(x, h)`：计算联合概率的对数 $\log p_\theta(x, h)$。
    这里的计算需要根据不同的假设，利用 $\log p_\theta(x|h)$ 和 $\log p(h)$ 来计算。我们可以假设 $h\sim p(h)$ 是一个离散的 Bernoulli 分布，或者 categorical 分布等。而 $p_\theta(x|h)$ 可以假设为高斯分布。
    当我们假设 $p_\theta(x|h)$ 是高斯分布时，我们使用 VAE 中的做法，即神经网络只输出均值，方差使用一个固定的超参数。在优化的时候使用 `MSE` 损失来优化均值。但是还需要使用联合概率 $\log p_\theta(x, h)$ 来接受拒绝采样。去掉常数后，我们最后返回的是 $\log p_\theta(x, h)= -\frac{\|x-\mu(x|h)\|_2^2}{2\sigma^2}$ 的值，其中 $\mu(x|h)$ 是神经网络输出的均值，$\sigma^2$ 是一个固定的超参数。
- 推断模型 `proposal_model` 需要提供的接口是：
  - `log_conditional_prob(h, x)`：计算条件概率的对数 $\log q_\phi(h|x)$。
    这里的计算也需要根据不同的假设，利用 $\log q_\phi(h|x)$ 来计算。我们可以假设 $h\sim q_\phi(h|x)$ 是一个离散的 Bernoulli 分布，或者 categorical 分布等。
- 另外，还需要一个采样的接口：
  - `sample_latent(x)`：从推理模型 $q_\phi(h|x)$ 中采样隐变量 $h$。
    这里的采样方法也需要根据不同的假设来实现。

需要一个 `MISampler`，用于从生成模型中采样隐变量 $h$。
`MISampler` 是否需要定义额外的类？还是就是一个函数？
我们从 MIS 采样的原理出发，其本质上是一个 MCMC 采样的方法，我们之前使用的 MCMC 方法类 （也就是 Langevin Dynamic）需要完成的逻辑是：给定能量函数，和迭代步骤，运行多次迭代，得到采样结果。需要存储步长。
$$
x_{\tau+1} = x_\tau - \frac{\epsilon^2}{2} \nabla_x U(x_\tau) + \epsilon \eta_\tau, \quad \eta_\tau \sim \mathcal{N}(0, I)
$$
而 MIS 采样的不同点是，不需要步长，只需要不断的使用模型提议，然后接受拒绝：

- Propose a new sample $h'$ from the proposal distribution $q_\phi(h|x)$.
- Calculate the acceptance ratio:
  $$
  A = \min\left(1, \frac{p_\theta(x, h') q_\phi(h|x)}{p_\theta(x, h) q_\phi(h'|x)}\right)
  $$
- Accept the new sample with probability $A$. If accepted, set $h = h'$; otherwise, keep the current sample $h$.

也就是说，我们既需要输入生成模型 `joint_model`，也需要输入推断模型 `proposal_model`，然后，我们还需要根据是否需要使用 cache 来决定是否存储之前的样本。

如果集成到 `LightningModule` 中的话，`MISampler` 可以作为 `JSA` 类的一个方法来实现。（这是比较方便的，因为我们可以在 `JSA` 类中存储 cache，逻辑都放到函数中实现）
如果不集成到 `LightningModule` 中的话，优势也是明显的，因为可以更加灵活的使用不同的采样方法。

最后为了方便实现，`MISampler` 独立于 `JSA` 类之外实现，但是作为 `JSA` 类的一个成员变量存在。

> 个人认为，为了方便，可以集成到 `JSA` 类中实现。但是，从更高的逻辑层面的角度，`JSA` 只需要一个采样器，来采到 `p_theta(h|x)` 的样本即可，采样器的具体实现可以有不同的方法。
> 需要验证一下，一个 model 传入到 `LightningModule` 和 `MISampler` 中，是否会同步更新参数。还是各自维护一份参数拷贝
> 以及，外面定义的 `model`，在使用 `DDP` 的时候，是会都保存一个副本，还是全局的。
> 上面的问题涉及到 python 的传参机制，python传参是对象的引用传递，因此传入参数后 `id` 是不变的，也就是说，里面的修改会影响到外面的对象。
> 只有遇到不可变对象，如 `int`，`tuple` 等，才会创建新的对象。
> 因此，传入 `model` 后，`MISampler` 和 `LightningModule` 里面的 `model` 应该是同一个对象，参数更新是同步的。我们在调用过程中，可以使用 `torch.no_grad()` 在 `MISampler` 里面禁用梯度计算，避免影响 `LightningModule` 的训练过程。

关于 cache 的问题，cache的大小应该是数据集的大小，每个数据点对应一个隐变量的样本。也就是说：

- 在初始化的时候，创建一个大小为数据集大小的 cache，存储每个数据点对应的隐变量样本。
- 在每次采样的时候，根据当前 batch 的数据点索引，从 cache 中取出对应的隐变量样本，作为初始值进行采样。
- 采样完成后，更新 cache 中对应的数据点的隐变量样本。

关键问题就是，这里的隐变量 $h$ 我们有怎样的假设？是 Bernoulli 还是 Categorical 还是其他的分布？

- K 维度的 Bernoulli 分布，表示每个隐变量是一个 K 维的二进制向量。则 cache 中存储的是一个大小为 (数据集大小, K) 的二进制矩阵。（每个样本为 `[0,1,0,1,1,1]` 类似的样本）
- K 维度的 Categorical 分布，表示每个隐变量是一个 K 类别的离散变量。则 cache 中存储的是一个大小为 (数据集大小,) 的整数向量。（每个样本为 `3`，`0`，`2` 类似的样本）

> K 维度的 Bernoulli 分布也可以变换为二进制编码的整数表示，例如 `[0,1,0,1]` 可以表示为 `5`。但是这样会增加编码和解码的复杂度，建议直接使用二进制矩阵表示。

## 数据集接口

数据集需要提供以下接口：
- `__len__()`：返回数据集的大小。
- `__getitem__(index)`：根据索引返回数据样本，需要注意的是，这里与一般的数据集不同，我们还需要返回数据点的索引，以便于在采样时更新 cache。

我们需要一个通用的数据集类，要求子类继承该类，并实现上述接口。（需要定义 `LightningDatasetModule`）

## 接口定义

### `Dataset`

定义一个抽象类 `JsaDataset`，继承自 `torch.utils.data.Dataset`，要求子类实现上述接口。

```python
from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset, DataLoader

class JsaDataset(Dataset):
    def __init__(self):
        super().__init__()
        # 初始化数据集

    def __len__(self):
        # 返回数据集大小
        pass

    def __getitem__(self, index):
        # 返回数据样本和索引
        pass

```

当我们需要使用新的数据集时，只需要继承 `JsaDataset`，并实现 `__len__` 和 `__getitem__` 方法即可。
而且在 `__getitem__` 方法中，除了返回数据样本外，还需要返回数据点的索引。

例如，我们有 `MNISTDataset`，继承自 `JsaDataset`，实现如下：

```python
from torchvision import datasets, transforms
from torch.utils.data import Dataset
class MNISTDataset(JsaDataset):
    def __init__(self, root: str, train: bool = True):
        super().__init__()
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            transform=transforms.ToTensor(),
            download=True,
        )

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, index):
        image, label = self.mnist[index]
        return image, label, index
```

### `DataModule`

定义一个 `JsaDataModule`，继承自 `LightningDataModule`，负责数据集的加载和划分。


```python
from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split

class JsaDataModule(LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        # 准备数据集，例如下载等
        pass

    def setup(self, stage=None):
        # 初始化数据集
        self.dataset = FullDataset()
        # 划分训练集和验证集
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [55000, 5000]
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    
```

### `MISampler` (Metropolis Independence Sampler)

首先需要定义一个抽象类 `BaseSampler`，定义采样器的基本接口。

```python
from abc import ABC, abstractmethod
class BaseSampler(ABC):
    @abstractmethod
    def step(self, x, h_old=None, idx=None):
      """ One step of sampling process

        Args:
            x: Input data
            h_old: Previous hidden variable sample (if any)
            idx: Index of the data point (for cache update)
        Returns:
            h_new: New hidden variable sample


      """
        pass
```

然后，我们定义 `MISampler`，继承自 `BaseSampler`，实现 Metropolis Independence Sampling 的逻辑。

```python
import torch
class MISampler(BaseSampler):
    def __init__(self, joint_model, proposal_model, use_cache=False, dataset_size=None):
        self.joint_model = joint_model
        self.proposal_model = proposal_model
        self.use_cache = use_cache
        if use_cache:
            assert dataset_size is not None, "Dataset size must be provided when using cache."
            self.cache = [None] * dataset_size  # Initialize cache
    
    @staticmethod
    def log_acceptance_ratio(p_theta, q_phi, x, h_new, h_old):
        pass

    def step(self, x, h_old=None, idx=None):
        pass

    # If needs to update cache
    def update_cache(self, idx, h_new):
      pass

    # If needs to multiple steps
    def sample(self, x, h_init=None, idx=None, num_steps=10):
      pass

    # If needs to save and load cache automatically when using checkpointing in the LightningModule
    def state_dict(self):
      pass

    def load_state_dict(self, state_dict):
      pass

    
```

### `JSA` LightningModule

同样地，我们定义 `JSA` 类，继承自 `LightningModule`，实现 JSA 方法的逻辑。

```python
from lightning.pytorch import LightningModule
class JSA(LightningModule):
    def __init__(self, joint_model, proposal_model, sampler: BaseSampler, lr: float = 1e-3):
        super().__init__()
        pass

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx, optimizer_idx):
        pass

    def configure_optimizers(self):
        pass
```

使用 `LightningCLI` 来简化训练过程。由于我们的 `JSA` 类中存在 `joint_model` 和 `proposal_model`，以及 `MISampler`，而后者的初始化需要调用前两个模型，因此我们需要使用 `hydra.utils.instantiate` 来实例化这些组件。

```python
from lightning.pytorch.cli import LightningCLI, ArgsType
from lightning.pytorch import LightningModule
from src.models.jsa import JSA
from src.data.mnist import MNISTDataset, JsaDataModule
from hydra.utils import instantiate

class JSA(LightningModule):
    def __init__(
        self,
        joint_model,
        proposal_model,
        sampler,
        lr_joint=1e-3,
        lr_proposal=1e-3,
        num_mis_steps=3,
        cache_start_epoch=0,
    ):
        super().__init__()
        self.joint_model: BaseJointModel = instantiate(joint_model)
        self.proposal_model: BaseProposalModel = instantiate(proposal_model)
        self.sampler: MISampler = instantiate(
            sampler,
            joint_model=self.joint_model,
            proposal_model=self.proposal_model,
        )
```

### `run_dataset.py` 脚本

最后，我们需要一个训练脚本 `run_dataset.py`，用于运行实验。

使用 `LightningCLI` 来简化训练过程。由于我们的 `JSA` 类中存在 `joint_model` 和 `proposal_model`，以及 `MISampler`，而后者的初始化需要调用前两个模型，因此我们需要使用 `hydra.utils.instantiate` 来实例化这些组件。

```python

def main():
    LightningCLI(
        run=True,
    )
        

if __name__ == "__main__":
    main()

```

需要使用的 `yaml` 文件如下所示：

```yaml
model:
  class_path: src.models.jsa.JSA
  init_args:
    joint_model:
      _target_: src.models.components.joint_model.JointModelBernoulliGaussian
      latent_dim: 200
      layers: [200, 200]
      output_dim: 784
      activation: leakyrelu
      final_activation: tanh
    proposal_model:
      _target_: src.models.components.proposal_model.ProposalModelBernoulli
      input_dim: 784
      latent_dim: 200
      layers: [200, 200]
      activation: leakyrelu
```
