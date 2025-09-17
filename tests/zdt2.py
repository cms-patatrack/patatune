import patatune
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

num_agents = 50
num_iterations = 250
num_params = 30

lb = [0.] * num_params
ub = [1.] * num_params
p_names = [f"x{i}" for i in range(num_params)]


def zdt2_objective(x):
    f1 = x[0]
    g = 1.0 + 9.0 * sum(x[1:]) / (len(x) - 1)
    h = 1.0 - np.power((f1 * 1.0 / g), 2)
    f2 = g * h
    return f1, f2

def zdt2_true_pareto(x):
    f1 = x
    f2 = 1.0 - np.power(x, 2)
    return f1, f2

patatune.Randomizer.rng = np.random.default_rng(42)
patatune.Logger.setLevel('DEBUG')
patatune.FileManager.working_dir = "tmp/zdt2/"
patatune.FileManager.loading_enabled = False
patatune.FileManager.loading_enabled = False
patatune.FileManager.saving_enabled = True
patatune.FileManager.saving_zarr_enabled = True
patatune.FileManager.saving_csv_enabled = True
patatune.FileManager.saving_pickle_enabled = False
patatune.FileManager.headers_enabled = True

objective = patatune.ElementWiseObjective(zdt2_objective, 2, objective_names=['f1', 'f2'], directions=['minimize', 'minimize'], true_pareto=zdt2_true_pareto)

pso = patatune.MOPSO(objective=objective, lower_bounds=lb, upper_bounds=ub, param_names=p_names,
                    num_particles=num_agents,
                    inertia_weight=0.5, cognitive_coefficient=1, social_coefficient=2,
                    initial_particles_position='random', topology='random', max_pareto_length=2*num_agents)

# run the optimization algorithm
pso.optimize(num_iterations)

fig, ax = plt.subplots()

pareto_front = pso.pareto_front
n_pareto_points = len(pareto_front)
pareto_x = [particle.fitness[0] for particle in pareto_front]
pareto_y = [particle.fitness[1] for particle in pareto_front]

real_x = (np.linspace(0, 1, n_pareto_points))
real_y = 1 - np.power(real_x, 2)
plt.scatter(real_x, real_y, s=5, c='red')
plt.scatter(pareto_x, pareto_y, s=5)

if not os.path.exists('tmp'):
    os.makedirs('tmp')
plt.savefig('tmp/pf.png')
