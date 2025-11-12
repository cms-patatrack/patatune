import patatune
import patatune.metrics
import numpy as np
import matplotlib.pyplot as plt

num_agents = 100
num_iterations = 100
num_params = 30

lb = [0.] * num_params
ub = [1.] * num_params

patatune.Logger.setLevel('DEBUG')

def zdt1_objective(x):
    f1 = x[0]
    g = 1 + 9.0 / (len(x)-1) * sum(x[1:])
    h = 1.0 - np.sqrt(f1 / g)
    f2 = g * h
    return f1, f2

def true_pareto(num_points):
    f1 = np.linspace(0, 1, num_points)
    f2 = 1-np.sqrt(f1)
    return np.array([f1, f2]).T
    
patatune.Randomizer.rng = np.random.default_rng(46)

patatune.FileManager.loading_enabled = False
patatune.FileManager.saving_enabled = False

objective = patatune.ElementWiseObjective(zdt1_objective, 2, true_pareto=true_pareto)

pso = patatune.MOPSO(objective=objective, lower_bounds=lb, upper_bounds=ub,
                      num_particles=num_agents,
                      inertia_weight=0.4, cognitive_coefficient=1.5, social_coefficient=2,
                      initial_particles_position='random', exploring_particles=True, max_pareto_length=2*num_agents)

# run the optimization algorithm
pso.optimize(num_iterations, max_iterations_without_improvement=5)

print("Generational distance: " ,pso.get_metric(patatune.metrics.generational_distance))
print("Inverted generational distance: " ,pso.get_metric(patatune.metrics.inverted_generational_distance))
print("Hypervolume: " ,pso.get_metric(patatune.metrics.hypervolume_indicator))

fig, ax = plt.subplots()
pareto_x = [particle.fitness[0] for particle in pso.pareto_front]
pareto_y = [particle.fitness[1] for particle in pso.pareto_front]
real_x, real_y = true_pareto(100).T
plt.scatter(pareto_x, pareto_y, s=5)
plt.scatter(real_x, real_y, s=5, c='red')
plt.show()