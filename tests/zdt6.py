import patatune
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import math

num_agents = 1000
num_iterations = 100
num_params = 10

lb = [0] * num_params
ub = [1] * num_params

patatune.Logger.setLevel('DEBUG')

def zdt6_objective1(x):
    return 1 - (np.exp(-4 * x[0]) * np.power(np.sin(6 * np.pi * x[0]), 6))


def zdt6_objective2(x):
    f1 = 1 - (np.exp(-4 * x[0]) * np.power(np.sin(6 * np.pi * x[0]), 6))
    g = 1 + 9 * np.power(sum(x[1:]) / (len(x) - 1), 0.25)
    h = 1.0 - (f1 / g)**2
    f2 = g * h
    return f2

patatune.FileManager.working_dir = "tmp/zdt6/"
patatune.FileManager.loading_enabled = False
patatune.FileManager.saving_enabled = False

if not os.path.exists(patatune.FileManager.working_dir):
    os.makedirs(patatune.FileManager.working_dir)

objective = patatune.ElementWiseObjective([zdt6_objective1, zdt6_objective2])

pso = patatune.MOPSO(objective=objective, lower_bounds=lb, upper_bounds=ub,
                      num_particles=num_agents,
                      inertia_weight=0.5, cognitive_coefficient=2, social_coefficient=0.5, initial_particles_position='random')

# run the optimization algorithm
pso.optimize(num_iterations)

fig, ax = plt.subplots()
pareto_front = pso.pareto_front
n_pareto_points = len(pareto_front)
pareto_x = [particle.fitness[0] for particle in pareto_front]
pareto_y = [particle.fitness[1] for particle in pareto_front]

real_x = np.linspace(0, 1, 100)
f1 = 1 - (np.exp(-4 * real_x) * np.power(np.sin(6 * np.pi * real_x), 6))
real_y = 1 - (f1 ** 2)

fig, ax = plt.subplots()
plt.scatter(pareto_x, pareto_y, s=5)
plt.scatter(real_x, real_y, s=5, c='red')

plt.savefig(patatune.FileManager.working_dir + 'pf.png')