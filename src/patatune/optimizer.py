"""Optimizer module for patatune.

This module contains the base Optimizer class that serves as a foundation for
various optimization algorithms implemented in the patatune package.
"""

class Optimizer:
    """Base class for optimization algorithms in patatune.
    The class that inherits from this one should implement `__init__`, `step`, and `optimize` methods.
    Raises a `NotImplementedError` if any of these methods are not implemented in the subclass.
    """
    def __init__(self) -> None:
        raise NotImplementedError

    def step(self):
        """Perform a single optimization step.
        This method should update the model parameters based on the optimization algorithm.
        """
        raise NotImplementedError

    def optimize(self):
        """Run the optimization process.
        This method should coordinate the optimization steps and handle any necessary
        setup or teardown for the optimization process.
        """
        raise NotImplementedError
