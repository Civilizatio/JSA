# scripts/train.py
from lightning.pytorch.cli import LightningCLI
import torch
import datetime

# torch.set_float32_matmul_precision("medium")

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from a .env file if present


class TimeStampedCLI(LightningCLI):
    def before_instantiate_classes(self):
        """Dynamically set the default save path with a timestamp."""

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        config = self.config[self.subcommand] if self.subcommand else self.config

        if "trainer" in config and "logger" in config["trainer"]:
            loggers = config["trainer"]["logger"]

            if not isinstance(loggers, list):
                loggers = [loggers]

            for logger in loggers:
                if isinstance(logger, (dict, object)) and hasattr(logger, "init_args"):
                    logger.init_args["version"] = timestamp


def main():
    torch.set_float32_matmul_precision("medium")

    TimeStampedCLI(
        run=True, 
        subclass_mode_model=True, 
        save_config_kwargs={"overwrite": True}
    )


if __name__ == "__main__":
    main()
