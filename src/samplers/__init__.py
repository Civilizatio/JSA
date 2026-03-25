"""Sampling algorithms for the JSA (Joint Stochastic Approximation) framework.

This module provides various Markov Chain Monte Carlo (MCMC) and stochastic samplers 
used to draw samples from the joint energy model components. Different samplers are 
tailored for different types of variables (e.g., continuous pixel-space variables vs. 
discrete categorical latents).

Available Samplers:
-------------------
* `LangevinSampler` : 
    Samples the continuous observation variable `x` given a fixed latent `h` 
    using overdamped Langevin dynamics driven by the distortion gradient.
* `NCGSampler` : 
    Norm-Constrained Gradient sampler. A gradient-guided Metropolis-Hastings 
    sampler designed for efficiently exploring discrete latent codes `h`.
* `MISampler` : 
    Multiple Importance Sampler (or related variants) for improving 
    proposal quality and escaping local modes.
* `JointNegativeSampler` : 
    Orchestrates the alternating block-coordinate sampling (e.g., alternating 
    between drawing `x` and `h`) to generate negative pairs for energy-based training.

To implement a custom sampler, please inherit from `src.base.base_sampler.BaseSampler`.
"""

from .joint_negative_sampler import JointNegativeSampler
from .langevin_sampler import LangevinSampler
from .misampler import MISampler
from .ncg_sampler import NCGSampler

__all__ = [
    "JointNegativeSampler",
    "LangevinSampler",
    "MISampler",
    "NCGSampler",
]