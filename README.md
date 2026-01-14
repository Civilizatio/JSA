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


## Running MNIST Experiments

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
tensorboard --logdir=egs/continuous_mnist/categorical_prior/version_11 --port=6006
```

``` bash

tensorboard --logdir=egs/cifar10/jsa/categorical_prior_conv/--port=6023
```

``` bash
tensorboard --logdir=egs/cifar10/vqgan/vq_gan_cifar10/2026-01-12_23-27-51 --port=6024
```
tensorboard --logdir=egs/continuous_mnist/categorical_prior_conv/base_version --port=6010

## Future development

- Add more experiments on different datasets and different models. Datasets like Fashion-MNIST, CIFAR-10, etc. Different models like VAE, GAN, etc.
