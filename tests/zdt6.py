import patatune
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import math

num_agents = 100
num_iterations = 300
num_params = 10

lb = [0] * num_params
ub = [1] * num_params

patatune.Logger.setLevel('DEBUG')

def zdt6_objective(x):
    f1 = 1 - (np.exp(-4 * x[0]) * np.power(np.sin(6 * np.pi * x[0]), 6))
    g = 1 + 9 * np.power(sum(x[1:]) / (len(x) - 1), 0.25)
    h = 1.0 - (f1 / g)**2
    f2 = g * h
    return f1, f2

def zdt6_true_pareto(x):
    f1 = x
    f2 = 1.0 - (x ** 2)
    return f1, f2

if not os.path.exists(patatune.FileManager.working_dir):
    os.makedirs(patatune.FileManager.working_dir)

lb = [0.] * num_params
ub = [1.] * num_params
p_names = [f"x{i}" for i in range(num_params)]

patatune.Randomizer.rng = np.random.default_rng(2)
patatune.Logger.setLevel('DEBUG')
patatune.FileManager.working_dir = f"tmp/zdt6/"
patatune.FileManager.loading_enabled = False
patatune.FileManager.saving_enabled = True
patatune.FileManager.saving_zarr_enabled = True
patatune.FileManager.saving_csv_enabled = True
patatune.FileManager.saving_pickle_enabled = False
patatune.FileManager.headers_enabled = True

objective = patatune.ElementWiseObjective(zdt6_objective, 2, objective_names=['f1', 'f2'], directions=['minimize', 'minimize'], true_pareto=zdt6_true_pareto)
pso = patatune.MOPSO(objective=objective, lower_bounds=lb, upper_bounds=ub, param_names=p_names,
                        num_particles=num_agents,
                        inertia_weight=0.75, cognitive_coefficient=2, social_coefficient=0.5,
                        initial_particles_position='random', topology='lower_weighted_crowding_distance', max_pareto_length=2*num_agents)

pso.optimize(num_iterations)

fig, ax = plt.subplots()
pareto_front = pso.pareto_front
n_pareto_points = len(pareto_front)
pareto_x = [particle.fitness[0] for particle in pareto_front]
pareto_y = [particle.fitness[1] for particle in pareto_front]

real_x, real_y = zdt6_true_pareto(np.linspace(0, 1, 100))

fig, ax = plt.subplots()
plt.scatter(real_x, real_y, s=5, c='red')
plt.scatter(pareto_x, pareto_y, s=5)

plt.savefig(patatune.FileManager.working_dir + 'pf.png')