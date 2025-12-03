# src/models/components/proposal_model.py
import torch
import torch.nn as nn

from src.base.base_jsa_modules import BaseProposalModel
from src.utils.mlp_utils import build_mlp


class ProposalModelBernoulli(BaseProposalModel):
    """q_phi(h|x)

    Must implement functions:
    - log_conditional_prob(h, x)
    - sample_latent(x)

    We assume Bernoulli distribution for q_phi(h|x)

    """

    def __init__(
        self, input_dim=784, layers=[512, 512], latent_dim=256, activation: str = "relu"
    ):
        super().__init__()

        self._latent_dim = latent_dim
        self.input_dim = input_dim

        self.net = build_mlp(
            input_dim=input_dim,
            layers=layers,
            output_dim=latent_dim,
            activation=activation,
        )

    @property
    def latent_dim(self):
        return self._latent_dim

    def sample_latent(self, x, num_samples=1):
        """Sample h ~ q(h|x)"""
        logits = self.net(x)  # [B, latent_dim]
        probs = torch.sigmoid(logits)  # [B, latent_dim]
        h_samples = torch.bernoulli(
            probs.repeat(num_samples, 1)  # [B * num_samples, latent_dim]
        )  # [B * num_samples, latent_dim]

        h_samples = (
            h_samples.view(num_samples, -1, self.latent_dim)
            .permute(1, 2, 0)
            .contiguous()
        )  # [B, latent_dim, num_samples]
        return h_samples.squeeze()  # [B, latent_dim, num_samples] if num_samples>1 else [B, latent_dim]

    def log_conditional_prob(self, h, x):
        logits = self.net(x)  # [B, latent_dim]
        return -torch.nn.functional.binary_cross_entropy_with_logits(
            logits, h, reduction="none"
        ).sum(dim=1)

    def forward(self, h, x):
        """Compute negative log conditional probability as loss

        Args:
            h: latent variables
            x: observed data

        Returns:
            loss: negative log conditional probability
        """
        log_cond = self.log_conditional_prob(h, x)
        return -log_cond.mean()


class ProposalModelCategorical(BaseProposalModel):
    """q_phi(h|x)

    Must implement functions:
    - log_conditional_prob(h, x)
    - sample_latent(x)

    We assume Categorical distribution for q_phi(h|x)



    """

    def __init__(
        self,
        input_dim=784,
        layers=[512, 512],
        num_latent_vars=10,
        num_categories=256,
        activation: str = "relu",
    ):
        """
        Args:
            input_dim: dimension of input x
            layers: list of hidden layer sizes
            num_latent_vars: number of latent variables (categories)
            num_categories: number of categories for each latent variable, integer or list
            activation: activation function to use in MLP
        """

        super().__init__()

        self.num_latent_vars = num_latent_vars

        if len(num_categories) == 1 and num_latent_vars > 1:
            self._num_categories = list(num_categories) * num_latent_vars
        else:
            assert len(num_categories) == num_latent_vars, "num_categories must be an integer or a list of length num_latent_vars"
            self._num_categories = list(num_categories)
        
        self.total_num_categories = sum(self._num_categories)

        self.net = build_mlp(
            input_dim=input_dim,
            layers=layers,
            output_dim=self.total_num_categories,
            activation=activation,
        )

    @property
    def latent_dim(self):
        return self.total_num_categories

    def sample_latent(self, x, num_samples=1, encoded=True):
        """Sample h ~ q(h|x)

        Args:
            x: Input tensor of shape [B, input_dim].
            num_samples: Number of samples to draw for each input.
            encoded: Whether to return encoded latent variables.
            
        Returns:
            h_samples: Tensor of shape [B, num_latent_vars, num_samples] containing sampled latent variable indices.

        here, each latent variable is represented as an integer index corresponding to the sampled category.

        """
        logits = self.net(x)  # [B, total_num_categories]
        
        split_logits = torch.split(logits, self._num_categories, dim=-1) # List of [B, num_categories_i]
        
        h_samples_list = []
        for i, logit in enumerate(split_logits):
            probs = torch.softmax(logit, dim=-1)  # [B, num_categories_i]
            h_samples = torch.multinomial(
                probs, num_samples=num_samples, replacement=True
            )  # [B, num_samples]
            h_samples = h_samples.view(
                -1, 1, num_samples
            )  # [B, 1, num_samples]
            h_samples_list.append(h_samples)
        h_samples = torch.cat(h_samples_list, dim=1)  # [B, num_latent_vars, num_samples], each entry is index of category
        if encoded:
            h_samples = self.encode_latent(h_samples, encoding_method="one_hot")  # [B, total_num_categories, num_samples]
        
        return h_samples.squeeze() # [B, total_num_categories, num_samples] if num_samples>1 else [B, total_num_categories]
            

    def encode_latent(self, h, encoding_method="one_hot"):
        """Encode the latent variable indices into one-hot or sinusoidal encoding.

        Args:
            h: Tensor of latent variable indices, shape [B, total_num_categories, num_samples].
            encoding: Encoding type, either "one_hot" or "sinusoidal".

        Returns:
            encoded_h: Encoded tensor, shape depends on the encoding type:
                - "one_hot": [B, num_latent_vars, num_samples, num_categories].
                - "sinusoidal": [B, num_latent_vars, num_samples, 2 * num_categories].

        Note:
            Here, we do not catcatenate along the latent variable dimension; each latent variable is encoded separately.
            This mechanism should be implemented in `JointModel`.
        """
        if encoding_method == "one_hot":
            encoded_h_list = []
            for i in range(self.num_latent_vars):
                h_i = h[:, i, :]  # [B, num_samples]
                h_i_one_hot = nn.functional.one_hot(
                    h_i.long(), num_classes=self._num_categories[i]
                ).float()  # [B, num_samples, num_categories_i]
                encoded_h_list.append(h_i_one_hot)  # [B, num_samples, num_categories_i]
            encoded_h = torch.cat(encoded_h_list, dim=-1).permute(0, 2, 1)  # [B, total_num_categories, num_samples]
        elif encoding_method == "sinusoidal":
            # Implement sinusoidal encoding if needed
            raise NotImplementedError("Sinusoidal encoding not implemented yet.")
        else:
            raise ValueError(f"Unknown encoding method: {encoding_method}")
        return encoded_h # [B, total_num_categories, num_samples]

    def log_conditional_prob(self, h, x):
        """Compute log q(h|x)

        Args:
            h: Tensor of latent variable indices, in one-hot encoding, shape [B, total_num_categories, num_samples].
            x: Input tensor of shape [B, input_dim].

        Returns:
            log_cond: Tensor of shape [B] containing log probabilities log q(h|x).

        For each latent variable, we compute the log probability of the sampled category given x.

        
        log q(h_i = h | x) = logits_{i, h} - log sum_{c} exp(logits_{i, c})
        

        Then sum over all latent variables to get log q(h|x).

        
        log q(h|x) = sum_{i=1}^{num_latent_vars} log q(h_i | x)
        
        """

        logits = self.net(x)  # [B, latent_dim * num_latent_vars]
        split_logits = torch.split(logits, self._num_categories, dim=-1)  # List of [B, num_categories_i]
        split_h = torch.split(h.squeeze(), self._num_categories, dim=-1)  # List of [B, num_categories_i]
        log_cond = 0
        for logit, h_i in zip(split_logits, split_h):
            # logit: [B, num_categories_i], h_i: [B, num_categories_i]
            logit_selected = torch.sum(
                logit * h_i, dim=-1
            )  # [B]
            log_sum_exp = torch.logsumexp(logit, dim=-1)  # [B]
            log_cond += logit_selected - log_sum_exp  # [B]
        return log_cond  # [B]

    def forward(self, h, x):
        """Compute negative log conditional probability as loss

        Args:
            h: latent variables
            x: observed data

        Returns:
            loss: negative log conditional probability
        """
        log_cond = self.log_conditional_prob(h, x)
        return -log_cond.mean()
