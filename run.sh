#!/bin/bash
# --config configs/jsa/categorical_prior_continuous_cifar10_conv.yaml \
# --config configs/vq_gan/cifar10.yaml \

CUDA_VISIBLE_DEVICES=2,3,4,5 PYTHONPATH=. python scripts/train.py fit --config configs/jsa/categorical_prior_continuous_cifar10_conv.yaml \
--trainer.devices=[0,1,2,3] \
--trainer.strategy=ddp_find_unused_parameters_true