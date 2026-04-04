#!/bin/bash

PYTHONPATH=. python scripts/train.py fit \
  --config configs/perceptual_jsa/jsa_base.yaml \
  --config configs/perceptual_jsa/jsa_stage_2.yaml \
  --trainer.devices 2,3