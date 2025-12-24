# src/utils/controllers.py

import math
import torch
import logging

logger = logging.getLogger(__name__)

class ParameterController:
    """ A generic controller to adjust a parameter based on a metric (e.g., acceptance rate).
    
    It increases or decreases the parameter to keep the metric within a target range.
    """
    def __init__(
        self,
        param_name: str,
        target_range: tuple[float, float],
        adjustment_rate: float = 0.01,
        min_val: float = 1e-6,
        max_val: float = 10.0,
        mode: str = "exponential",  # 'exponential' or 'linear'
        direction: str = "inverse", # 'direct' or 'inverse' relationship
    ):
        """
        Args:
            param_name: Name of the parameter (for logging).
            target_range: (min_target, max_target). No adjustment if metric is inside.
            adjustment_rate: How fast to adjust (learning rate for the parameter).
            min_val: Minimum allowed value for the parameter.
            max_val: Maximum allowed value for the parameter.
            mode: 'exponential' (multiplicative update) or 'linear' (additive update).
            direction: 
                'inverse': metric > target -> decrease param (e.g., sigma vs accept_rate).
                'direct': metric > target -> increase param.
        """
        self.param_name = param_name
        self.min_target, self.max_target = target_range
        self.adjustment_rate = adjustment_rate
        self.min_val = min_val
        self.max_val = max_val
        self.mode = mode
        self.direction = direction
        
        assert self.min_target <= self.max_target
        assert self.direction in ["direct", "inverse"]

    def step(self, current_val: float, metric_val: float) -> tuple[float, dict]:
        """
        Calculate the new parameter value based on the metric.
        
        Returns:
            new_val: The updated parameter value.
            info: A dictionary containing debug info (scale_factor, diff, etc.).
        """
        diff = 0.0
        
        # 1. Calculate deviation
        if metric_val < self.min_target:
            # Metric is too low
            diff = self.min_target - metric_val # Positive diff
        elif metric_val > self.max_target:
            # Metric is too high
            diff = self.max_target - metric_val # Negative diff
        else:
            # Within range, no change
            return current_val, {"scale_factor": 1.0, "diff": 0.0, "status": "stable"}

        # 2. Determine adjustment direction
        # If direction is 'inverse' (like sigma vs accept_rate):
        #   metric too low (diff > 0) -> need to INCREASE param -> factor > 1
        #   metric too high (diff < 0) -> need to DECREASE param -> factor < 1
        # If direction is 'direct':
        #   metric too low (diff > 0) -> need to DECREASE param -> factor < 1
        sign = 1.0 if self.direction == "inverse" else -1.0
        
        # 3. Calculate update
        new_val = current_val
        scale_factor = 1.0
        
        if self.mode == "exponential":
            # new = old * exp(rate * diff * sign)
            scale_factor = math.exp(self.adjustment_rate * diff * sign)
            new_val = current_val * scale_factor
        elif self.mode == "linear":
            # new = old + (rate * diff * sign)
            delta = self.adjustment_rate * diff * sign
            new_val = current_val + delta
            scale_factor = 1.0 # Not applicable for linear

        # 4. Clip
        new_val = max(self.min_val, min(self.max_val, new_val))
        
        info = {
            "scale_factor": scale_factor,
            "diff": diff,
            "metric": metric_val,
            "old_val": current_val,
            "new_val": new_val,
            "status": "adjusting"
        }
        
        return new_val, info