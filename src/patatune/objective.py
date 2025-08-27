import numpy as np


class Objective():
    def __init__(self, objective_functions, num_objectives=None, objective_names=None ,true_pareto=None) -> None:
        if not isinstance(objective_functions, list):
            self.objective_functions = [objective_functions]
        else:
            self.objective_functions = objective_functions

        if num_objectives is None:
            self.num_objectives = len(self.objective_functions)
        else:
            self.num_objectives = num_objectives
        
        if objective_names is None:
            self.objective_names = [f"objective_{i}" for i in range(self.num_objectives)]
        else:
            if len(objective_names) != self.num_objectives:
                raise ValueError(
                    f"Number of objective names ({len(objective_names)}) does not match number of objectives ({self.num_objectives}).")
            self.objective_names = objective_names

        self.true_pareto = true_pareto

    def evaluate(self, items):
        result = [objective_function(items)
                  for objective_function in self.objective_functions]
        solutions = []
        for r in result:
            if len(np.shape(r)) > 1:
                for sub_r in r:
                    solutions.append(sub_r)
            else:
                solutions.append(r)
        return np.array(solutions)

    def type(self):
        return self.__class__.__name__


class ElementWiseObjective(Objective):
    def evaluate(self, items):
        result = [[obj_func(item) for item in items]
                  for obj_func in self.objective_functions]
        solutions = []
        for r in result:
            if len(np.shape(r)) > 1:
                for sub_r in (np.array(r).T):
                    solutions.append(sub_r)
            else:
                solutions.append(r)
        return np.array(solutions).T


class BatchObjective(Objective):
    pass
