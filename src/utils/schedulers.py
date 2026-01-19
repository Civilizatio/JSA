# src/utils/schedulers.py

import math


class SigmaScheduler:
    """Scheduler that anneals sigma from max_val to min_val.

    Strategy:
        - Warmup phase: sigma ~= max_val
        - Decay phase: smooth decay to min_val
    """

    def __init__(
        self,
        max_val: float,
        min_val: float,
        warmup_steps: int,
        total_steps: int,
        hold_steps: int = 0,
        mode: str = "cosine",  # "linear" | "cosine" | "exponential"
        eps: float = 1e-8,
    ):
        assert total_steps > warmup_steps
        self.max_val = max_val
        self.min_val = min_val
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.hold_steps = hold_steps
        self.mode = mode
        self.eps = eps
        
        self.decay_steps = max(1, total_steps - warmup_steps - hold_steps)
        
        self.last_step = -1
        self.current_sigma = max_val
        

    def get_sigma(self, step: int) -> float:
        """Return scheduled sigma at given step."""

        # 1. Warmup phase
        if step < self.warmup_steps:
            return self.max_val

        # 2. Decay phase & Hold phase
        t = (step - self.warmup_steps) / float(self.decay_steps)
        t = min(max(t, 0.0), 1.0) # if step>warmup+decay, t=1.0

        if self.mode == "linear":
            sigma = self.max_val * (1 - t) + self.min_val * t

        elif self.mode == "cosine":
            sigma = self.min_val + 0.5 * (self.max_val - self.min_val) * (
                1 + math.cos(math.pi * t)
            )

        elif self.mode == "exponential":
            # smooth exponential decay
            log_max = math.log(self.max_val + self.eps)
            log_min = math.log(self.min_val + self.eps)
            sigma = math.exp(log_max * (1 - t) + log_min * t)

        else:
            raise ValueError(f"Unknown scheduler mode: {self.mode}")

        return float(sigma)
