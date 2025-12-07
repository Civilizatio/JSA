# src/base/base_sampler.py
from abc import ABC, abstractmethod
from torch.nn import Module

class BaseSampler(ABC, Module):
    """Base sampler for JSA framework"""

    @abstractmethod
    def step(self, x, h_old=None, idx=None):
        """Perform a sampling step

        Args:
            x: Input data
            h_old: Previous latent sample (if any)
            idx: Index of the data point (for cache usage)

        Returns:
            h_new: New sample
            log_q: Log probability of the new sample
        """
        pass
    
    @abstractmethod
    def sample(self, x, h_init=None, idx=None, num_steps=1):
        """ Multi-step sampling to generate multiple samples

        Args:
            x: Input data
            h_init: Initial latent sample (if any)
            idx: Index of the data point (for cache usage)
            num_steps: Number of sampling steps to perform

        Returns:
            samples: Generated samples
            log_qs: Log probabilities of the samples
        """
        pass
