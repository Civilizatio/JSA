#!/bin/bash
# --config configs/jsa/categorical_prior_continuous_cifar10_conv.yaml \
# --config configs/vq_gan/cifar10.yaml \

PYTHONPATH=. python scripts/train.py fit --config configs/jsa/categorical_prior_continuous_cifar10_conv.yaml