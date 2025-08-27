import patatune
import numpy as np
import matplotlib.pyplot as plt


num_agents = 500
num_iterations = 200
num_params = 2

lb = [0., 0]
ub = [10., 5]

default_point = [3., 2]


def always_true(x):
    return True


objective = patatune.Objective([always_true])


pso = patatune.MOPSO(objective=objective, lower_bounds=lb, upper_bounds=ub, num_particles=num_agents,
                      inertia_weight=0.6, cognitive_coefficient=1, social_coefficient=2, initial_particles_position='gaussian', default_point=default_point)

fig, ax = plt.subplots()
plt.scatter([p.position[0] for p in pso.particles], [p.position[1]
            for p in pso.particles], s=5)
plt.savefig('tmp/initial.png')
