import patatune

lb = [0., False]
ub = [2., True]

patatune.FileManager.working_dir = "tmp/bool/"
patatune.FileManager.saving_enabled = True


def func(x):
    if x[1] == False:
        print(x, x[0] >= 1)
        return int(x[0] >= 1)
    else:
        print(x, x[0] < 1)
        return int(x[0] < 1)


objective = patatune.ElementWiseObjective([func])

pso = patatune.MOPSO(objective=objective, lower_bounds=lb, upper_bounds=ub,
                      initial_particles_position='random', num_particles=5)

pso.optimize(10)
