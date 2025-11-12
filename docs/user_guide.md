Beyond what has been introduced in the [Example](index.md#example), PATATUNE allows to fine tune the optimization process for the user.

This involves the definition of the parameters to optimize, the definition of the optimization function, and the configuration of the optimization algorithms.

In addition, several utilities are implemented to adapt to the user environment and workflow.

The following documentation details the functionalities available and how to use them.

## Parameters definition

PATATUNE uses the lower and upper bound of the parameters to define the search space.

In addition it deducts the parameters type from the bounds.

One can define the parameters bounds as two lists:

```python
lb = [0, 0., False]
ub = [5, 5., True]
```
This define a parameter space with three parameters, where:

 - the first parameter is an integer going from 0 (included) to 5 (included);
 - the second parameter is a floating point value going from 0 (included) to 5 (excluded);
 - the third and last parameter is a boolean value that can either be `True` or `False`.
  
When passed to the optimization algorithm, PATATUNE will check that the lenght of the two lists is equal, warning the user in case of mismatch and using the lowest range.  
It will then check the types of the variable, throwing a warning in case of mismatch and using the most permissive type (`bool` < `int` < `float`).


## Objective function definition

The user can use any function as objective function to evaluate in the optimization process.

For each set of parameters, identifying a 'position' in the search space, the objective function will return a value, also called 'fitness'.

PATATUNE can optimize the parameters against any number of objective functions.

Two different methods can be identified in defining the functions:

- Objective functions that implements a method to evaluate a list of positions at once, returning a corresponding list of fitnesses
- Objective functions that implements a method to evaluate a single position at a time, returning the fitness of the single position

PATATUNE allows to define these objective functions through the `Objective` class and it's subclasses.

During the optimization, the `evaulate` function of the class will be called behaving differently based on its implementation

### Objective

The [Objective][patatune.objective.Objective] class is the base class for defining objective functions and takes as argument a list of objective functions `[f1, f2, ...]`.


In the [evaluate][patatune.objective.Objective.evaluate] method, all objective functions are executed as:

```python
[f(positions) for f in objective_functions]
```

Each objective function is run once per iteration on all particle positions simultaneously.

**Input format**: Each objective function receives a list of arrays with shape `(num_particles, num_parameters)`, where each row represents a position in the search space.

**Output format**: Returns an array with shape `(num_particles, num_objectives)`, where each row contains the evaluated objective values for a particle.

This approach is useful when:

- Objective functions implements their own way to handle multiple set of parameters
- Batch processing is implemented externally
- All particles can be evaluated simultaneously

Example usage:

```python
def f1(x):
    return x[:, 0]**2

def f2(x):
    return (x[:, 0] - 2)**2

objective = patatune.Objective([f1, f2])
```

### ElementWise Objective

The [ElementWiseObjective][patatune.objective.ElementWiseObjective] class inherits from `Objective` and provides a way to evaluate objective functions element-wise, one particle at a time.

Unlike the base `Objective` class where each function receives all positions at once, `ElementWiseObjective` calls each objective function individually for every particle position.

The [evaluate][patatune.objective.ElementWiseObjective.evaluate] method iterates over each position and applies the objective functions:

```python
[[f(position) for position in positions] for f in objective_functions]
```

This approach is useful when:

- The objective function is designed to work on a single parameter set at a time
- The evaluation will be vectorized by python
- Evaluations are independent and don't benefit from batch processing

For example, defining a simple element-wise objective:

```python
def f1(x):
    return x[0]**2

def f2(x):
    g = 1 + 9.0 / (len(x)-1) * sum(x[1:])
    h = 1.0 - np.sqrt(f1 / g)
    return g * h

objective = patatune.ElementWiseObjective([f1, f2])
```

The objective function receives a single parameter array `x` and returns a tuple of objective values `(f1, f2)`.

### Asynchronous Objective evaluation

PATATUNE provides two classes for asynchronous objective function evaluation, enabling efficient parallel processing when dealing with computationally expensive evaluations or external services.

#### AsyncElementWiseObjective

The [AsyncElementWiseObjective][patatune.objective.AsyncElementWiseObjective] class evaluates objective functions asynchronously on each particle independently. This approach is useful when:

- Each evaluation is expensive and independent
- Evaluations can benefit from concurrent execution (I/O-bound operations, API calls, etc.)
- You want maximum parallelism without manual batching

All objective functions must be defined with `async def` and the class automatically handles concurrent execution using `asyncio.gather`.

Example usage:

```python
async def async_objective_function(x):
    f1 = x[0]
    g = 1 + 9.0 / (len(x)-1) * sum(x[1:])
    h = 1.0 - np.sqrt(f1 / g)
    f2 = g * h
    return [f1, f2]

objective = patatune.AsyncElementWiseObjective(async_objective_function)
```

#### BatchObjective

The [BatchObjective][patatune.objective.BatchObjective] class provides asynchronous batch evaluation, where particles are grouped into batches before evaluation. This is particularly useful when:

- The evaluation system works more efficiently with batches
- You want to control resource consumption by limiting concurrent operations
- External systems have rate limits or batch processing capabilities

The `BatchObjective` requires:

- **Asynchronous objective functions**: Functions must be defined with `async def`
- **Batch size**: Parameter that controls how many particles are evaluated in each batch

Example usage:

```python
async def batched_evaluation(params):
    # params is a list of parameter sets (one batch)
    results = []
    for p in params:
        f1 = 4 * p[0]**2 + 4 * p[1]**2
        f2 = (p[0] - 5)**2 + (p[1] - 5)**2
        results.append([f1, f2])
    return results

objective = patatune.BatchObjective(
    [batched_evaluation],
    batch_size=10
)
```

The `BatchObjective` automatically splits the particle positions into batches of the specified size and evaluates them concurrently using `asyncio.gather`.

### Multiple objectives definition

The class determines the number of objectives based on the length of the list of objective_functions passed as argument, assuming that a single objective value is evaluated by each function.

However, in case an objective function were to return more than one value, the user can specify the number of expected objectives returned with the optional `num_objectives` argument.

Optionally the user can pass the names of the objectives in the `objective_names` argument, that will be used by the [FileManager][patatune.util.FileManager] when saving the results of the optimization.
If they are not passed as arguments, they default to `['objective_0','objective_1',...]`.

Finally, the user can pass a callable in the `true_pareto` argument.
This is a function that will return a list of points of size equal to the archive of optimal solution obtained after the optimization, with the fitnesses of each point.

The argument is completely optional and used in measuring the [GD][patatune.metrics.generational_distance] and [IGD][patatune.metrics.inverted_generational_distance] metrics.

```python
def zdt1(x):
    f1 = x[0]
    g = 1 + 9.0 / (len(x)-1) * sum(x[1:])
    h = 1.0 - np.sqrt(f1 / g)
    f2 = g * h
    return [f1, f2]

def true_pareto(num_points):
    f1 = np.linspace(0, 1, num_points)
    f2 = 1 - np.sqrt(f1)
    return np.array([f1, f2]).T

objective = patatune.ElementWiseObjective(zdt1, num_objectives=2, objective_names=['f1', 'f2'], true_pareto=true_pareto)
```

#### Defining the direction of the optimization

By default, each objective is optimized to be minimized. To override this behaviour, the user can pass the `directions` argument as a list of strings (i.e. `['minimize', 'maximize', 'minimize']`), listing the optimization direction for each objective.
If the number of objectives don't match the lenght of the strings, PATATUNE raises an error.

```python
objective = patatune.ElementWiseObjective([efficiency_function, fake_rate_function], directions=['maximize', 'minimize'])
```

## Otimization algorithm configuration

The [Optimizer][patatune.optimizer.Optimizer] base class allows to define custom multi-objective optimization algorithm to be used in the same way by the user.

Currently the library implements a Multi-Objective Particle Swarm Optimization (MOPSO) algorithm.

### MOPSO

The MOPSO algorithm is a versatile optimization tool designed for solving multi-objective problems. It leverages the concept of swarm to navigate the search space and find optimal solutions.

#### Algorithm Overview

The implementation in the library is close to the one defined [here](https://doi.org/10.1109/TEVC.2004.826067):

 - A swarm of particle is initialized in the parameters space
 - The objective functions are evaluated for each particle
 - Each particle is tested for [dominance][patatune.util.get_dominated]
 - The dominant particles are added to the archive of optimal solutions
 - Each particle updates its velocity and position based on its local best and a global best chosen from the archive
 - The process is repeated for a given number of iterations
 - At the end of the optimization, the archive of optimal solutions is returne

#### Basic Configuration

The MOPSO class can be configured through several parameters:

- **Objective**: MOPSO can optimize virtually any objective function defined by the user as an instance of the [Objective][patatune.objective.Objective] class or its subclasses.
- **Boundary Constraints**: Users has to define the lower and upper bounds of the [parameters](#parameters-definition) to optimize, and the name of the parameters.
- **Swarm Size**: The user can define the number of particles in the swarm with the `num_particles` parameter.

```python
mopso = patatune.MOPSO(
    objective=objective,
    lower_bounds=[0.0] * 30,
    upper_bounds=[1.0] * 30,
    num_particles=100
)
```

#### Hyperparameters

The behavior of the MOPSO algorithm can be fine-tuned through three hyperparameters:

- **Inertia Weight**: Control the inertia of the particles, influencing their tendency to continue moving in the same direction.
- **Cognitive Coefficients**: Control the influence of the particle's own best-known position on its movement.
- **Social Coefficients**: Control the influence of the swarm's best-known position on the particle's movement.

Higher inertia weight will lead to particles maintaining their velocity, promoting exploration of the search space, while lower inertia weight will encourage particles to focus on their local best solutions.

Higher value of the social coefficient will lead the particles to be more attracted towards the global best solution found by the swarm, promoting exploration but leading to potentially lower diversity in the solutions.

Higher value of the cognitive coefficient will lead to a more exploitative behavior, meaning that the particles will be more likely to refine their search in the vicinity of known good solutions, potentially leading to faster convergence but risking getting stuck in local optima.

#### Topology Strategies

The choice of the `global_best` particle from the archive can be configured through the `topology` parameter:

 - `random`: the `global_best` is chosen randomly from the archive
 - `round_robin`: the `global_best` is chosen in round robin fashion from the archive
 - `lower_weighted_crowding_distance` and `higher_weighted_crowding_distance`: the `global_best` is chosen based on the crowding distance of the particles in the archive, favoring less crowded areas or more crowded areas respectively.

Using the crowding distance can help to maintain diversity in the solutions found by the swarm, preventing premature convergence to a single solution, however it is computationally more expensive.  
Using the random or round robin strategies is computationally cheaper, but can lead to less diverse solutions.

#### Setting the initial particle positions

The initial position of the particles can be defined through the `initial_particle_position` parameter and the `default_point` parameter:

  - `random` uniform distribution
  - `gaussian` distribution around the `default_point`
  - all in the `lower_bounds` or `upper_bounds` of the parameter space

Setting the `default_point` parameter with any option other than `gaussian` will ensure that at least one particle starts from that position.
The gaussian distribution will center the particles around the `default_point`, with a standard deviation equal to one fourth of the distance between the lower and upper bounds.
The particles will be clamped to stay within the defined bounds.

The choice of the initial position can influence the convergence speed and the variety of solutions found by the swarm.

```python
mopso = patatune.MOPSO(
    objective=objective,
    lower_bounds=[0.0] * 30,
    upper_bounds=[1.0] * 30,
    num_particles=100,
    initial_particle_position='gaussian',
    default_point=[0.5] * 30
)
```

#### Additional Features

- **Exploration Mode**: An optional exploration mode enables particles to scatter from their position when they don't improve for a given number of iterations

Exploration mode can help to escape local optima and explore new areas of the search space, potentially leading to better overall solutions.
However it's implementation is still experimental and should be used with caution.

- **Limit on Archive Size**: The archive of optimal solutions can be limited in size, removing the most crowded solutions when the limit fixed with the `max_pareto_length` is reached.


See the [API reference][patatune.mopso.mopso.MOPSO] for additional information on the parameters.

#### Running the optimization

MOPSO can be run using the `optimize` method for a specific number of iterations, or it can also be run interactively by calling the `step` function to perform a single iteration.

```python
mopso.optimize(num_iterations=200)
```

If the `exploring_particles` option is enabled, the user can pass the `max_iterations_without_improvement` parameter to define after how many iterations without improvement a particle should be scattered in the search space.

```python
mopso.optimize(num_iterations=200, max_iterations_without_improvement=10)
```

## Utilities

PATATUNE provides several utilities to adapt to the user workflow and environment.

### File Manager

The [FileManager][patatune.util.FileManager] class provides functionalities to manage file saving and loading during the optimization process.

If the `FileManager.saving_enabled` flag is set to `True`, the state of the optimizer will be saved in the working directory specified in the `FileManager.working_directory`.

The `FileManager` will create the directory if it does not exist.

The `FileManager` supports different file formats for saving the state of the optimizer, that can be enabled or disabled through the corresponding flags `saving_pickle_enabled`, `saving_json_enabled`, `saving_csv_enabled`, and `saving_zarr_enabled`.

For example, in `MOPSO`, if the correct flags are set, the following files will be created in the working directory at each saving step:

- a `checkpoint/pareto_front.csv` file containing the current archive of optimal solutions with the parameters and objective values in floating point representation with 18 decimals, with comma delimiter. If `FileManager.headers_enabled` is set to `True`, a header row will be included with the [parameter names](#parameters-definition) and [objective names](#multiple-objectives-definition).
- a `checkpoint/individual_states.csv` file containing the current state of each particle in the swarm, with the parameters and velocities in floating point representation with 18 decimals, with comma delimiter. If `FileManager.headers_enabled` is set to `True`, a header row will be included with the [parameter names](#parameters-definition) and velocity names (`velocity_<parameter_name>`).
- a `checkpoint/mopso.pkl` file containing the full MOPSO object serialized with `dill`.

At the end of all iterations, a `checkpoint/mopso.zip` file will be created containing the history of the optimization process saved at each saving step as a zarr archive.
The archive will contain a group for each iteration named `iteration_<iteration_number>`, containing a dataset `data` with the parameters and fitnesses of all particles at that iteration, and a group `pareto_front` containing a dataset `data` with the parameters and fitnesses of the Pareto front particles.
Along with the datasets, the archive will contain attributes with the parameter names, objective names, lower bounds and upper bounds of the optimization process.

If the `FileManager.loading_enabled` flag is set to `True`, the optimizer will attempt to load its state from the working directory at the beginning of the optimization process, using the latest saved `checkpoint/mopso.pkl` file.

This allows to resume an optimization process from the last saved state.


### Random

PATATUNE relies on random number generation. To make sure to obtain reproducible results an helper function allows to set the seed for every random generation performed by the algorithm:

```python
patatune.Randomizer.rng = np.random.default_rng(42)
```

### Logging

You can configure the amount of logging information printed on terminal by passing a string to the [setLevel][logging.Logger.setLevel] function of the `patatune.Logger`:

```python
patatune.Logger.setLevel('DEBUG')
```

The supported levels - from least to most verbose - are: `'ERROR'`, `'WARNING'`, `'INFO'`, `'DEBUG'`
