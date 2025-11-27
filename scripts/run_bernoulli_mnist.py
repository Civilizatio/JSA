# scripts/run_bernoulli_MNIST.py
from lightning.pytorch.cli import LightningCLI
from src.models.jsa import JSA
from src.data.mnist import MNISTDataModule
from hydra.utils import instantiate
import logging
import torch
torch.set_float32_matmul_precision("medium")


from dotenv import load_dotenv
import os
load_dotenv()


def main():
    LightningCLI(run=True)


if __name__ == "__main__":
    main()
