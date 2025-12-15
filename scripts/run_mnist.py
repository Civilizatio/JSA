# scripts/run_mnist.py
from lightning.pytorch.cli import LightningCLI
import torch
import sys

torch.set_float32_matmul_precision("medium")

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from a .env file if present


class JSA_CLI(LightningCLI):
    def before_instantiate_classes(self):
        """ Validate model configuration parameters before instantiation. 
        
        """

        config = self.config

        subcommand = config.subcommand
        if subcommand not in ["fit", "validate", "test", "predict"]:
            return

        cfg = config.get(subcommand)

        if "model" in cfg and "init_args" in cfg.model:

            init_args = cfg.model.init_args

            joint_args = init_args.get("joint_model", {})
            proposal_args = init_args.get("proposal_model", {})

            if joint_args and proposal_args:

                # 1. joint_model.output_dim should match proposal_model.input_dim
                proposal_input_dim = proposal_args.get("input_dim", None)
                joint_output_dim = joint_args.get("output_dim", None)
                if proposal_input_dim != joint_output_dim:
                    print(
                        f"Parameter mismatch: joint_model.output_dim ({joint_output_dim}) != proposal_model.input_dim ({proposal_input_dim})."
                    )
                    sys.exit(1)

                # 2. proposal_model.num_latent_vars should match joint_model.num_latent_vars
                joint_num_latent_vars = joint_args.get("num_latent_vars", None)
                proposal_num_latent_vars = proposal_args.get("num_latent_vars", None)
                if joint_num_latent_vars != proposal_num_latent_vars:
                    print(
                        f"Parameter mismatch: joint_model.num_latent_vars ({joint_num_latent_vars}) != proposal_model.num_latent_vars ({proposal_num_latent_vars})."
                    )
                    sys.exit(1)

                # 3. proposal_model.num_categories should match joint_model.num_categories (if applicable)
                joint_num_categories = joint_args.get("num_categories", None)
                proposal_num_categories = proposal_args.get("num_categories", None)
                if joint_num_categories != proposal_num_categories:
                    print(
                        f"Parameter mismatch: joint_model.num_categories ({joint_num_categories}) != proposal_model.num_categories ({proposal_num_categories})."
                    )
                    sys.exit(1)


def main():
    JSA_CLI(run=True)


if __name__ == "__main__":
    main()
