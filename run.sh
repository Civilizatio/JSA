#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. python scripts/train.py fit --config ./configs/vq_gan/cifar10.yaml \
--trainer.devices=[0,1,2,3] \
--trainer.strategy=ddp_find_unused_parameters_true