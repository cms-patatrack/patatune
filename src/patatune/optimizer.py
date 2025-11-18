"""Optimizer module for patatune.

This module contains the base [`Optimizer`][patatune.optimizer.Optimizer] class that other optimizers in
``patatune`` should inherit from.
"""

class Optimizer:
    """Base class for optimization algorithms.

    Subclasses must implement `__init__`,
    [`step`][patatune.optimizer.Optimizer.step], and
    [`optimize`][patatune.optimizer.Optimizer.optimize].

    Raises:
        NotImplementedError: If methods are not implemented by subclasses.
    """
    def __init__(self) -> None:
        raise NotImplementedError

    def step(self):
        """Perform a single optimization step.
        
        Subclasses should update internal state (particles/parameters) in this
        method.
        """
        raise NotImplementedError

    def optimize(self):
        """Run the optimization loop.

        This method should coordinate repeated calls to :meth:`step` and any
        required setup/teardown.
        """
        raise NotImplementedError
