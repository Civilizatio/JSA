# JSA

Codes for reproducing experiments  in “Zhijian Ou, Yunfu Song. Joint Stochastic Approximation and its Application to Learning Discrete Latent Variable Models, UAI 2020”.

## Prerequistes

- Python 3.6
- PyTorch 1.0
- TensorFlow 1.8

## Usage

- For the generative modeling with Bernoulli variables experiment, please refer to **bernoulli_MNIST** folder.
- For the generative modeling with categorical variables  experiment, please refer to **categorical_MNIST** folder.
- For the structured output prediction  experiment, please refer to **structured_prediction** folder.

## Acknowledgments

The categorical MNIST experiments are based on the implementation [ARSM](https://github.com/ARM-gradient/ARSM).


## Running Bernoulli MNIST Experiments

训练指令：

```bash
PYTHONPATH=. python scripts/run_bernoulli_mnist.py fit --config ./configs/categorical_prior_continuous_mnist.yaml
```

从 `checkpoints` 恢复训练指令：

```bash
PYTHONPATH=. python scripts/run_bernoulli_mnist.py fit \
            --config ./configs/bernoulli_prior_binary_mnist.yaml \
            --ckpt_path ./egs/bernoulli_mnist/binary_mnist/version_4/checkpoints/best-checkpoint.ckpt

```

查看 TensorBoard 日志：

``` bash
tensorboard --logdir=egs/continuous_mnist/categorical_prior/version_6 --port=6034
```