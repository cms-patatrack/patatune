import patatune
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

num_agents = 100
num_iterations = 200
num_params = 11

lb = [0] * num_params
ub = [1073741825] * num_params


def zdt5_objective1(x):
    return 1 + u(x[0])


def zdt5_objective2(x):
    f1 = 1 + u(x[0])
    g = sum([v(u(i)) for i in x[1:]])
    h = 1.0 / f1
    f2 = g * h
    return f2

def u(x):
    c = 0
    while x:
        c += 1
        x &= x - 1
    return c


def v(x):
    un = u(x)
    if un < 5:
        return 2 + un
    elif un == 5:
        return 1
    else:
        return 0


patatune.FileManager.working_dir = "tmp/zdt5/"
patatune.FileManager.loading_enabled = False
patatune.FileManager.saving_enabled = False

if not os.path.exists(patatune.FileManager.working_dir):
    os.makedirs(patatune.FileManager.working_dir)

objective = patatune.ElementWiseObjective([zdt5_objective1, zdt5_objective2])

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

real_x = np.array([u(x) for x in np.linspace(0, 1073741825, n_pareto_points, dtype=np.int_)]) + 1
real_y = [10 / (1 + u(x)) for x in real_x]
plt.scatter(real_x, real_y, s=5, c='red')
plt.scatter(pareto_x, pareto_y, s=5)

plt.savefig(patatune.FileManager.working_dir + 'pf.png')