import patatune

lb1 = [0., 1, 0., 5.]
lb2 = [0., 1, 0., "lower"]
ub1 = [1, 2, 3, 6]
ub2 = [2., 2, 1.]


print([type(lb) for lb in lb1])
print([type(lb) for lb in ub1])


def always_true(x):
    return True


objective = patatune.Objective([always_true])

pso = patatune.MOPSO(objective=objective, lower_bounds=lb1,
                      upper_bounds=ub1, initial_particles_position='random')

print([type(lb) for lb in pso.lower_bounds])
print([type(ub) for ub in pso.upper_bounds])

pso = patatune.MOPSO(objective=objective, lower_bounds=lb1,
                      upper_bounds=ub2, initial_particles_position='random')

try:
    pso = patatune.MOPSO(objective=objective,
                          lower_bounds=lb2, upper_bounds=ub1)
except ValueError as e:
    print(e)

print([type(lb) for lb in pso.lower_bounds])
print([type(ub) for ub in pso.upper_bounds])

lbt = [0., 0, False]
ubt = [1., 5, True]

psot = patatune.MOPSO(objective=objective, lower_bounds=lbt,
                       upper_bounds=ubt, initial_particles_position='spread')

print("\n".join([f"{p.position}" for p in psot.particles]))
