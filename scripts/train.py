# scripts/train.py
from lightning.pytorch.cli import LightningCLI
import torch
import sys
import inspect

torch.set_float32_matmul_precision("medium")

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from a .env file if present


def main():

    LightningCLI(
        run=True, subclass_mode_model=True, save_config_kwargs={"overwrite": True}
    )


if __name__ == "__main__":
    main()
