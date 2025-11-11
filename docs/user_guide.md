## Objective Function

Depending on the optimization mode, the objective function can be defined in two way:

1. As `Objective`, the objective function is evaluated once per iteration and is called as:

    ```python
    f([particle.position for particle in self.particles])
    ```

    the argument of the optimization function is a list of arrays: an array of parameters for each particle.  
    The output is a list of fitnesses: the value(s) to minimize for each particle.

2. As `ElementWiseObjective`, the objective function is evaluated particle by particle at every iteration and is called as:

    ```python
    f(self.position) # self is a Particle object
    ```

    the argument of the optimization function is an array of elements corresponding to the parameters to optimize.  
    The output is the fitness of the particle: the value(s) to minimize in order to solve the optimization problem.  

See the `tests` and `examples` folders for examples.

## MOPSO

The Multi-Objective Particle Swarm Optimization (MOPSO) algorithm is a versatile optimization tool designed for solving multi-objective problems. It leverages the concept of swarm to navigate the search space and find optimal solutions.

- **Objective**: MOPSO can optimize virtually any objective function defined by the user.
- **Boundary Constraints**: Users can specify lower and upper bounds for each parameter, and uses the definition of the boundaries to detect the variable types (i.e. `0.0` for floating points, `0` for integers, `False` for booleans)
- **Swarm Size**: Adjusting the number of particles in the swarm allows to balance convergence speed and computation intensity.
- **Inertia Weight**: Control the inertia of the particle velocity to influence the global and local search capabilities.
- **Cognitive and Social Coefficients**: Fine-tune the cognitive and social components of the velocity update equation to steer the search process.
- **Initial Particle Position**: Offers multiple strategies for initializing particle positions:

  - `random` uniform distribution
  - `gaussian` distribution around a given point
  - all in the `lower_bounds` or `upper_bounds` of the parameter space
- **Exploration Mode**: An optional exploration mode enables particles to scatter from their position when they don't improve for a given number of iterations
- **Swarm Topology**: Supports different swarm topologies, affecting how particles chose the `global_best` to follow.

See the docstring for additional information on the parameters.

## Checkpoint system

patatune can be run using the `optimize` method for a specific number of iterations, or it can also be run interactively by calling the `step` function to perform a single iteration.

In addition patatune allows to stop the execution, saving the state, and restore the execution from the leftover run.

To do this, first enable saving and enabling using the `FileManager` helper class:

```python
patatune.FileManager.working_dir = "tmp/zdt1/"
patatune.FileManager.loading_enabled = True
patatune.FileManager.saving_enabled = True
```

After launching `optimize` the state of patatune will be saved in the `mopso.pkl` file inside the working directory.

A new run of the script will attempt to load the file and restart the execution from the iteration it was stopped at.

For example, if you run optimize until iteration 100, save and then rerun till iteration 200, patatune will call the step function for iteration 101 to 200.

The saving option allow also to export the state of the particles in every iteration inside a `history` directory in the working directory.

## Random

The MOPSO patatune heavily relies on randomnumber generation. To make sure to obtain reproducible results an helper function allows to set the seed for every random generation performed by the algortihm:

```python
patatune.Randomizer.rng = np.random.default_rng(42)
```

## Logging

You can configure the amount of logging information printed on terminal with:

```python
patatune.Logger.setLevel('DEBUG')
```

The supported levels - from least to most verbose - are: `ERROR`, `WARN`, `INFO`, `DEBUG`
