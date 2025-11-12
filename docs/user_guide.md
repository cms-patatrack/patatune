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
def batch_evaluation(positions):
    results = []
    for x in positions:
        f1 = x[0]**2
        f2 = (x[0] - 2)**2
        results.append([f1, f2])
    return results

objective = patatune.Objective(
    [batch_evaluation],
    num_objectives=2,
    objective_names=['f1', 'f2'],
    directions=['minimize', 'minimize']
)

mopso = patatune.MOPSO(
    objective=objective,
    lower_bounds=[-10.0],
    upper_bounds=[10.0],
    num_particles=50
)
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
def objective_function(x):
    f1 = x[0]
    g = 1 + 9.0 / (len(x)-1) * sum(x[1:])
    h = 1.0 - np.sqrt(f1 / g)
    f2 = g * h
    return f1, f2

objective = patatune.ElementWiseObjective(
    objective_function, 
    num_objectives=2,
    objective_names=['f1', 'f2'],
    directions=['minimize', 'minimize']
)

mopso = patatune.MOPSO(
    objective=objective,
    lower_bounds=[0.0] * 30,
    upper_bounds=[1.0] * 30,
    num_particles=100
)
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
    return f1, f2

objective = patatune.AsyncElementWiseObjective(
    async_objective_function,
    num_objectives=2,
    objective_names=['f1', 'f2'],
    directions=['minimize', 'minimize']
)

mopso = patatune.MOPSO(
    objective=objective,
    lower_bounds=[0.0] * 30,
    upper_bounds=[1.0] * 30,
    num_particles=100
)
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
    batch_size=10,
    num_objectives=2,
    objective_names=['f1', 'f2'],
    directions=['minimize', 'minimize']
)

mopso = patatune.MOPSO(
    objective=objective,
    lower_bounds=[0.0, 0.0],
    upper_bounds=[5.0, 3.0],
    num_particles=100
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

#### Defining the direction of the optimization

By default, each objective is optimized to be minimized. To override this behaviour, the user can pass the `directions` argument as a list of strings (i.e. `['minimize', 'maximize', 'minimize']`), listing the optimization direction for each objective.
If the number of objectives don't match the lenght of the strings, PATATUNE raises an error.

## Otimization algorithm configuration

### MOPSO

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

## Utilities

### Checkpoint system

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

### Random

The MOPSO patatune heavily relies on randomnumber generation. To make sure to obtain reproducible results an helper function allows to set the seed for every random generation performed by the algortihm:

```python
patatune.Randomizer.rng = np.random.default_rng(42)
```

### Logging

You can configure the amount of logging information printed on terminal by passing a string to the [setLevel][logging.Logger.setLevel] function of the `patatune.Logger`:

```python
patatune.Logger.setLevel('DEBUG')
```

The supported levels - from least to most verbose - are: `'ERROR'`, `'WARNING'`, `'INFO'`, `'DEBUG'`
