"""
Learning-rate schedulers used by UWIR training.
GradualWarmupScheduler, CosineAnnealingRestartLR, CosineAnnealingRestartCyclicLR
are unchanged from the upstream implementation and are domain-agnostic.
"""

import math

from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler

# ---------------------------------------------------------------------------
# Gradual warm-up wrapper
# ---------------------------------------------------------------------------


class GradualWarmupScheduler(_LRScheduler):
    """
    Linearly ramps the learning rate from 0 (or base_lr) up to
    base_lr * multiplier over `total_epoch` epochs, then hands off to
    `after_scheduler`.

    Reference: "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour".
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        if multiplier < 1.0:
            raise ValueError("multiplier must be ≥ 1.0")
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (self.last_epoch / self.total_epoch) for base_lr in self.base_lrs]
        return [
            base_lr * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
            for base_lr in self.base_lrs
        ]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [
                base_lr * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr, strict=True):
                param_group["lr"] = lr
        else:
            self.after_scheduler.step(metrics, None if epoch is None else epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if not isinstance(self.after_scheduler, ReduceLROnPlateau):
            if self.finished and self.after_scheduler:
                self.after_scheduler.step(None if epoch is None else epoch - self.total_epoch)
            else:
                super().step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_position_from_periods(iteration, cumulative_period):
    """Return the index of the right-closest period boundary."""
    for i, period in enumerate(cumulative_period):
        if iteration <= period:
            return i
    return len(cumulative_period) - 1


# ---------------------------------------------------------------------------
# Cyclic cosine-annealing with restarts
# ---------------------------------------------------------------------------


class CosineAnnealingRestartCyclicLR(_LRScheduler):
    """
    Cosine annealing with multiple independently-weighted restart cycles.
    Each cycle can have its own period, restart weight, and eta_min.

    Args:
        periods        (list[int])   : Length of each cycle in epochs.
        restart_weights(list[float]) : LR scale at the start of each cycle.
        eta_mins       (list[float]) : Minimum LR for each cycle.
    """

    def __init__(self, optimizer, periods, restart_weights=(1,), eta_mins=(0,), last_epoch=-1):
        assert len(periods) == len(restart_weights), (
            "periods and restart_weights must have the same length"
        )
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_mins = eta_mins
        self.cumulative_period = [sum(periods[: i + 1]) for i in range(len(periods))]
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        idx = _get_position_from_periods(self.last_epoch, self.cumulative_period)
        weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        period = self.periods[idx]
        eta_min = self.eta_mins[idx]
        progress = (self.last_epoch - nearest_restart) / period
        return [
            eta_min + weight * 0.5 * (base_lr - eta_min) * (1 + math.cos(math.pi * progress))
            for base_lr in self.base_lrs
        ]


# ---------------------------------------------------------------------------
# Standard cosine-annealing with restarts
# ---------------------------------------------------------------------------


class CosineAnnealingRestartLR(_LRScheduler):
    """
    Cosine annealing with periodic restarts and a single global eta_min.

    Args:
        periods        (list[int])   : Length of each restart cycle.
        restart_weights(list[float]) : LR scale at the start of each cycle.
        eta_min        (float)       : Global minimum learning rate.
    """

    def __init__(self, optimizer, periods, restart_weights=(1,), eta_min=0, last_epoch=-1):
        assert len(periods) == len(restart_weights), (
            "periods and restart_weights must have the same length"
        )
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_min = eta_min
        self.cumulative_period = [sum(periods[: i + 1]) for i in range(len(periods))]
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        idx = _get_position_from_periods(self.last_epoch, self.cumulative_period)
        weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        period = self.periods[idx]
        progress = (self.last_epoch - nearest_restart) / period
        return [
            self.eta_min
            + weight * 0.5 * (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress))
            for base_lr in self.base_lrs
        ]
