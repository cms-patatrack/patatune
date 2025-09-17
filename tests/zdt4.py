import patatune
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

num_agents = 100
num_iterations = 600
num_params = 10

lb = [0.] + [-5.] * (num_params - 1)
ub = [1.] + [5.] * (num_params - 1)
p_names = [f"x{i}" for i in range(num_params)]

def zdt4_objective(x):
    f1 = x[0]
    g = 1.0 + 10 * (len(x) - 1) + sum([i**2 - 10 * np.cos(4 * np.pi * i) for i in x[1:]])
    h = 1.0 - np.sqrt(f1 / g)
    f2 = g * h
    return f1, f2

def zdt4_true_pareto(x):
    f1 = x
    f2 = 1.0 - np.sqrt(x)
    return f1, f2

patatune.Randomizer.rng = np.random.default_rng(42)
patatune.Logger.setLevel('INFO')
patatune.FileManager.working_dir = f"tmp/zdt4"
patatune.FileManager.loading_enabled = False
patatune.FileManager.saving_enabled = True
patatune.FileManager.saving_zarr_enabled = True
patatune.FileManager.saving_csv_enabled = True
patatune.FileManager.saving_pickle_enabled = False
patatune.FileManager.headers_enabled = True

objective = patatune.ElementWiseObjective(zdt4_objective, 2, objective_names=['f1', 'f2'], directions=['minimize', 'minimize'], true_pareto=zdt4_true_pareto)
pso = patatune.MOPSO(objective=objective, lower_bounds=lb, upper_bounds=ub, param_names=p_names,
                        num_particles=num_agents,
                        inertia_weight=0.5, cognitive_coefficient=1.5, social_coefficient=2,
                        initial_particles_position='random', topology='random', max_pareto_length=2*num_agents)

pso.optimize(num_iterations)
print(f"Optimization completed. Pareto front size: {len(pso.pareto_front)}")


fig, ax = plt.subplots()

pareto_front = pso.pareto_front
n_pareto_points = len(pareto_front)
pareto_x = [particle.fitness[0] for particle in pareto_front]
pareto_y = [particle.fitness[1] for particle in pareto_front]

real_x = (np.linspace(0, 1, 100))
real_y = 1 - np.sqrt(real_x)
plt.scatter(real_x, real_y, s=5, c='red')
plt.scatter(pareto_x, pareto_y, s=5)

plt.savefig(patatune.FileManager.working_dir + 'pf.png')