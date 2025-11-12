"""Objective classes for multi-objective optimization.

Provides a base [`Objective`][patatune.objective.Objective] and concrete implementations for
element-wise, batched and asynchronous evaluations.
"""

import numpy as np
import asyncio

class Objective():
    """Base class for defining objective functions.

    The class is used to evaluate multiple objective functions for all particles simultaneously.
    The classes that inherit from this one should implement the `evaluate` method.
    
    Args:
        objective_functions (list | callable): A list of callables (or a single
            callable) that compute objective values.
        num_objectives (int, optional): Number of objectives. Defaults to
            ``len(objective_functions)``.
        directions (list[str], optional): List of 'minimize' or 'maximize' for
            each objective. Defaults to all 'minimize'.
        objective_names (list[str], optional): Names for each objective.
        true_pareto (callable, optional): Function that returns the true Pareto front. Takes as input
            the number of points and returns a 2D array of shape `(num_points, num_objectives)`.
    """
    def __init__(self, objective_functions, num_objectives=None, directions=None, objective_names=None ,true_pareto=None) -> None:
        if not isinstance(objective_functions, list):
            self.objective_functions = [objective_functions]
        else:
            self.objective_functions = objective_functions

        if num_objectives is None:
            self.num_objectives = len(self.objective_functions)
        else:
            self.num_objectives = num_objectives
        
        if directions is None:
            self.directions = [1] * self.num_objectives
        else:
            if len(directions) != self.num_objectives:
                raise ValueError(
                    f"Number of directions ({len(directions)}) does not match number of objectives ({self.num_objectives}).")
            self.directions = []
            for direction in directions:
                if direction not in ['minimize', 'maximize']:
                    raise ValueError(
                        f"Direction must be either 'minimize' or 'maximize', got '{direction}'.")
                self.directions.append(1 if direction == 'minimize' else -1)

        if objective_names is None:
            self.objective_names = [f"objective_{i}" for i in range(self.num_objectives)]
        else:
            if len(objective_names) != self.num_objectives:
                raise ValueError(
                    f"Number of objective names ({len(objective_names)}) does not match number of objectives ({self.num_objectives}).")
            self.objective_names = objective_names

        self.true_pareto = true_pareto

    def evaluate(self, items):
        """Passes all items to each objective function and collects the results.
        
        Args:
            items (list): List of parameter sets to evaluate with shape ``(num_particles, num_parameters)``.

        Returns:
            (np.ndarray): Array of shape (num_particles, num_objectives) with evaluated objective values.
        """
        result = [objective_function(items)
                  for objective_function in self.objective_functions]
        solutions = []
        for r in result:
            if len(np.shape(r)) > 1:
                for sub_r in r:
                    solutions.append(sub_r)
            else:
                solutions.append(r)
        return np.array(solutions) * self.directions

    def type(self):
        return self.__class__.__name__


class ElementWiseObjective(Objective):
    """Element-wise objective class.
    
    Inherits from the base Objective class and implements the evaluate method
    to evaluate each objective function on individual items.
    """
    def evaluate(self, items):
        """Passes each item one by one to each objective function and collects the results.

        Args:
            items (list): List of parameter sets to evaluate of shape (num_particles, num_parameters).

        Returns:
            (np.ndarray): Array of shape (num_particles, num_objectives) with evaluated objective values.
        """
        result = [[obj_func(item) for item in items]
                  for obj_func in self.objective_functions]
        solutions = []
        for r in result:
            if len(np.shape(r)) > 1:
                for sub_r in (np.array(r).T):
                    solutions.append(sub_r)
            else:
                solutions.append(r)
        solutions = np.array(solutions).T * self.directions
        return solutions

class BatchObjective(Objective):
    """Batch objective class.

    Inherits from the base Objective class and implements the evaluate method
    to evaluate each objective function on batches of items asynchronously.
    
    
     Attributes:
        batch_size (int): Size of each batch for evaluation.
    """
    def __init__(self, objective_functions, batch_size, num_objectives=None, directions=None, objective_names=None, true_pareto=None):
        super().__init__(objective_functions, num_objectives, directions, objective_names, true_pareto)
        self.batch_size = batch_size

    async def _async_evaluate(self, obj_func, batches):
        if not asyncio.iscoroutinefunction(obj_func):
            raise ValueError(f"Objective function {obj_func} must be asynchronous for BatchObjective.")
        tasks = [obj_func(batch) for batch in batches]
        return await asyncio.gather(*tasks)

    def evaluate(self, items):
        """Passes items in batches to each objective function asynchronously and collects the results.

        Args:
            items (list): List of parameter sets to evaluate of shape (num_particles, num_parameters).

        Returns:
            (np.ndarray): Array of shape (num_particles, num_objectives) with evaluated objective values.
        """
        if len(items) % self.batch_size == 0:
            batches = [items[i:i + self.batch_size]
                       for i in range(0, len(items), self.batch_size)]
        else:
            batches = [items[i:i + self.batch_size]
                       for i in range(0, len(items) - len(items) % self.batch_size, self.batch_size)]
            batches.append(items[len(items) - len(items) % self.batch_size:])
        
        result = []
        for obj_func in self.objective_functions:
            if not asyncio.iscoroutinefunction(obj_func):
                raise ValueError("All objective functions must be asynchronous for BatchObjective.")
            tasks_output = asyncio.run(self._async_evaluate(obj_func, batches))
            result.append(np.concatenate(tasks_output))

        solutions = []
        for r in result:
            if len(np.shape(r)) > 1:
                for sub_r in r:
                    solutions.append(sub_r)
            else:
                solutions.append(r)
        return np.array(solutions) * self.directions

class AsyncElementWiseObjective(Objective):
    """Asynchronous element-wise objective class.
    """
    async def _async_evaluate(self, obj_func, items):
        if not asyncio.iscoroutinefunction(obj_func):
            raise ValueError(f"Objective function {obj_func} must be asynchronous.")
        tasks = [obj_func(item) for item in items]
        return await asyncio.gather(*tasks)

    def evaluate(self, items):
        """Passes each item one by one to each asynchronous objective function and collects the results.
        
        Args:
            items (list): List of parameter sets to evaluate of shape (num_particles, num_parameters).
        
        Returns:
            (np.ndarray): Array of shape (num_particles, num_objectives) with evaluated objective values.
        """
        result = []
        for obj_func in self.objective_functions:
            tasks_output = asyncio.run(self._async_evaluate(obj_func, items))
            result.append(np.array(tasks_output))
        solutions = []
        for r in result:
            if len(np.shape(r)) > 1:
                for sub_r in (np.array(r).T):
                    solutions.append(sub_r)
            else:
                solutions.append(r)
        solutions = np.array(solutions).T * self.directions
        return solutions