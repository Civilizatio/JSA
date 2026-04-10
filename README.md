# JSA

Codes for reproducing experiments  in “Zhijian Ou, Yunfu Song. Joint Stochastic Approximation and its Application to Learning Discrete Latent Variable Models, UAI 2020”.

Recoded by Ke Li for better modularity and extensibility.

## Prerequistes

- Python 3.12
- PyTorch 2.9.1
- Install other dependencies by running:

```bash
pip install -r requirements.txt
```

## Usage

代码的具体框架和接口可见 [structure.md](./docs/structure.md)。


## Running Experiments

> 需要的环境变量可以定义在 `.env` 文件中，例如可以定义 `PYTHONPATH=.`。则运行下面指令时可以不需要前面加 `PYTHONPATH=.`。

训练入口在 `scripts/train.py`。
训练指令：

```bash
PYTHONPATH=. python scripts/train.py fit \
    --config ./configs/categorical_prior_continuous_mnist_conv.yaml
```

从 `checkpoints` 恢复训练指令：

```bash
PYTHONPATH=. python scripts/train.py fit \
    --config ./configs/categorical_prior_continuous_cifar10_conv.yaml \
    --ckpt_path ./egs/continuous_cifar10/categorical_prior_conv/version_0/checkpoints/best-checkpoint.ckpt \
            
```

Test:

```bash
PYTHONPATH=. python scripts/train.py test \
    --config ./configs/categorical_prior_continuous_mnist.yaml \
    --ckpt_path ./egs/continuous_mnist/categorical_prior/version_3/checkpoints/best-checkpoint.ckpt \
    --trainer.devices=[0] \
    --trainer.strategy=auto \

```
```bash
PYTHONPATH=. python scripts/train.py fit --config ./configs/jsa/categorical_prior_continuous_cifar10_conv.yaml
```



查看 TensorBoard 日志：

``` bash
tensorboard --logdir=egs/cifar10/perceptual_jsa/cifar10_stage1_decoder/2026-04-08_10-37-27 --port=6024
```

``` bash

tensorboard --logdir=egs/cifar10/perceptual_jsa/cifar10_stage1_decoder/2026-04-09_09-51-28 --port 6023
```

``` bash
tensorboard --logdir=egs/cifar10/perceptual_jsa/cifar10_stage1_decoder/2026-04-08_10-39-28 --port=6025
```
tensorboard --logdir=egs/cifar10/perceptual_jsa/cifar10_stage1_decoder/2026-04-09_15-00-25 --port=6013

## Running Perceptual Version

后面新增了一个基于感知损失的版本，其原理可以参考 [jsa_perceptual_loss.md](./docs/jsa_perceptual_loss.md)。分为三个阶段。

### Stage 1: Train the decoder

```bash
PYTHONPATH=. python scripts/train.py fit \
  --config configs/perceptual_jsa/jsa_base.yaml \
  --config configs/perceptual_jsa/jsa_stage_1.yaml
```

### Stage 2: Train the prior with the decoder fixed

```bash
PYTHONPATH=. python scripts/train.py fit \
  --config configs/perceptual_jsa/jsa_base.yaml \
  --config configs/perceptual_jsa/jsa_stage_2.yaml \
  --model.init_args.init_from_ckpt=/path/to/stage1.ckpt
```

### Stage 3: Jointly fine-tune the prior and decoder

```bash
PYTHONPATH=. python scripts/train.py fit \
  --config configs/perceptual_jsa/jsa_base.yaml \
  --config configs/perceptual_jsa/jsa_stage_3.yaml \
  --model.init_args.init_from_ckpt=/path/to/stage2.ckpt
```

## Future development

- Add more experiments on different datasets and different models. Datasets like Fashion-MNIST, CIFAR-10, etc. Different models like VAE, GAN, etc.
