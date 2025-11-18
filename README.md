![PATATUNE](https://raw.githubusercontent.com/cms-patatrack/PATATUNE/refs/heads/docs/docs/Patatune.png)

*A Framework for Metaheuristic Multi-Objective Optimization for High Energy Physics*

![PyPI - Version](https://img.shields.io/pypi/v/patatune)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://cms-patatrack.github.io/patatune/)

---

**Documentation:** [https://cms-patatrack.github.io/patatune](https://cms-patatrack.github.io/patatune)

**Source code:** [https://github.com/cms-patatrack/patatune](https://github.com/cms-patatrack/patatune)

---

PATATUNE is a Python package that provides a framework for multi-objective optimization algorithms, including the Multi-Objective Particle Swarm Optimization (MOPSO) method.
Its primary purpose is to automate the optimization of the parameters of user-defined functions.
The package has been developed with the needs of CMS and Patatrack in mind.

The key features are:

- **Easy to use** and learn.
- **Pluggable** Multi-objective optimization model with Multi-Objective Particle Swarm Optimization implemented via the `MOPSO` class.
- Multiple objective definition supported, for any **user-defined objective** function.
- Support for different **parameter types** (int, float, bool).
- Built-in **metrics** for convergence/quality assessment: Generational Distance, Inverted GD, Hypervolume.
- Persistence and **checkpointing** via `FileManager` (save/load pickle, CSV, Zarr); supports resuming runs and per-iteration history export.

## Installation

PATATUNE is available on [PyPi](https://pypi.org/project/patatune/).

To install it you can simply run:

```bash
pip install patatune
```

If you want to use the latest development version on the main branch:

1. Clone this repository
2. Navigate into the project directory
3. Install the package and its dependencies using pip:

    ```bash
    pip install .
    ```
You can install a project in “editable” or “develop” mode while you’re working on it. When installed as editable, a project can be edited in-place without reinstallation:

```bash
pip install -e .
```

## Requirements

PATATUNE is written for Python 3.9+ and depends on a small set of scientific Python packages.  The following are required to run the library:

- [numpy](https://numpy.org/doc/)
- [dill](https://dill.readthedocs.io/en/latest/)

Optional functionality is provided by extras:

- [numba](https://numba.pydata.org/numba-doc/dev/index.html) (optional — JIT acceleration)
- [zarr==2.*](https://zarr.readthedocs.io/en/v2.2.0/) (optional — save/load history in Zarr format)

These dependencies are declared in `pyproject.toml` and can be installed with pip (see Installation below). If you need the optional extras, install with `extra`:

```bash
pip install patatune[extra]
```

The additional example require additional libraries ([matplotlib](https://matplotlib.org/stable/index.html), [pandas](https://pandas.pydata.org/docs/)). If you want to run them, install with `tests`:

```bash
pip install patatune[tests]
```
or, to include every optional dependecy:

```bash
pip install patatune[all]
```

## Example

Currently the package provides the `patatune` module that defines an optimization algorithm: `MOPSO`.
PATATUNE relies on a few helper classes to handle configuration and the objective functions. To use this module in your Python projects:

1. Import the required modules:

    ```python
    import patatune
    ```

2. Define the objective function to be optimized. i.e.:

    ```python
    def f1(x):
        return 4 * x[0]**2 + 4 * x[1]**2


    def f2(x):
        return (x[0] - 5)**2 + (x[1] - 5)**2

    objectives = patatune.ElementWiseObjective([f1, f2])
    ```

3. Define the boundaries of the parameters:

    ```python
    lb = [0.0, 0.0]
    ub = [5.0, 3.0]
    ```

4. Create the MOPSO object with the configuration of the algorithm

    ```python
    mopso = patatune.MOPSO( objectives,
                            lower_bounds=lb, upper_bounds=ub,
                            num_particles=50,
                            inertia_weight=0.4, cognitive_coefficient=1.5, social_coefficient=2)
    ```

5. Run the optimization algorithm

    ```python
    pareto = mopso.optimize(num_iterations = 100)
    ```

The output will be the archive of optimal solutions found by the algorithm after 100 iterations.

The output is a Python list containing Particle objects (instances of `patatune.mopso.particle.Particle`). 
You can easily extract a compact representation from the returned list. For example:

```python
for p in pareto:
    print("position:", p.position, "fitness:", p.fitness)
```

Example printed output:

```
id: 0  position: [1.8 1.6] fitness: [ 23.5 21.6]
id: 49 position: [0.0 0.0] fitness: [ 0.   50. ]
id: 18 position: [1.0 1.0] fitness: [ 8.3  31.7]
id: 29 position: [5.0 3.0] fitness: [ 136. 4.  ]
id: 16 position: [0.0 0.0] fitness: [ 0.   50. ]
...

```

## Contributing

Contributions are welcome. If you want to contribute, please follow the [Contribution guidelines](https://github.com/cms-patatrack/PATATUNE/blob/main/CONTRIBUTING.md).

## License

PATATUNE is distributed under the [MPL 2.0 License](https://github.com/cms-patatrack/PATATUNE/blob/main/LICENSE). Feel free to use, modify, and distribute the code following the terms of the license.  
