import patatune
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation


lb = [-10.0]
ub = [10.0]

num_agents = 100
num_iterations = 10
num_params = 1


def f1(x):
    return x**2


def f2(x):
    return (x - 2) ** 2


def f(params):
    return [[f1(params[i][0]), f2(params[i][0])] for i in range(len(params))]


patatune.FileManager.working_dir = "tmp/schaffer/"
patatune.FileManager.loading_enabled = True

objective = patatune.Objective([f])

pso = patatune.MOPSO(objective=objective, lower_bounds=lb, upper_bounds=ub,
                      num_particles=num_agents,
                      inertia_weight=0.5, cognitive_coefficient=1, social_coefficient=1)

# run the optimization algorithm
pso.optimize()

metrics = [pd.read_csv('tmp/schaffer/history/iteration' + str(i) + '.csv',
                       header=None,
                       usecols=[num_params, num_params + 1]).transpose().to_numpy()
           for i in range(num_iterations)]


fig, ax = plt.subplots()


def animate(i):
    fig.clear()
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)
    ax.scatter(metrics[i][0], metrics[i][1], c='red', s=10)
    ax.set_title(str(i))


ani = animation.FuncAnimation(
    fig, animate, interval=200, frames=range(num_iterations))
ani.save('tmp/schaffer/checkpoint/metrics.gif', writer='pillow')
