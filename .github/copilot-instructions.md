## Quick orientation for AI coding assistants

Be concise and repository-specific. This project is a Python package under `src/patatune` that implements
Multi-Objective Particle Swarm Optimization (MOPSO) and helper utilities. Key source files:

- `src/patatune/mopso/mopso.py` — main algorithm implementation (class `MOPSO`).
- `src/patatune/mopso/particle.py` — particle representation and topology strategies.
- `src/patatune/objective.py` — `Objective`, `ElementWiseObjective`, `BatchObjective`, `AsyncElementWiseObjective` APIs.
- `src/patatune/metrics.py` — metrics (GD, IGD, hypervolume) and `get_metric` usage in `MOPSO`.
- `src/patatune/optimizer.py` — optimizer base class (abstract contract for `step`/`optimize`).
- `src/patatune/util.py` — I/O (FileManager), `Randomizer`, `Logger`, and utility functions referenced widely.
- `example/` and `tests/` — runnable examples and tests; use these for real usage patterns.

What matters (concrete patterns)
- Two objective function styles are supported:
  - `Objective`: functions that accept the full list of particle positions (batch mode).
    Signature example: `f([particle.position for particle in self.particles]) -> array_like`.
  - `ElementWiseObjective`: element-wise call per particle. Signature: `f(self.position) -> scalar or tuple`.
  Use `ElementWiseObjective` for simple per-particle functions and `BatchObjective` for async/batched evaluation.

- Parameter typing is inferred from `lower_bounds` / `upper_bounds` (int, float, bool). Keep bounds length consistent.
- Initialization options used by `MOPSO`: `random`, `gaussian`, `lower_bounds`, `upper_bounds`.
- Topology strings: `random`, `lower_weighted_crowding_distance`, `higher_weighted_crowding_distance`, `round_robin`.

Persistence and reproducibility
- The code relies on `patatune.FileManager` for checkpoints: `save_pickle`, `load_pickle`, `save_csv`, `save_zarr`.
  Example pattern: set `patatune.FileManager.loading_enabled = True` to resume runs.
- Random seeds: set `patatune.Randomizer.rng = np.random.default_rng(SEED)` for deterministic runs.

Dev/build/test notes (discoverable)
- Install for development: `python3 -m pip install -e .` (package uses `src/` layout; see `pyproject.toml`).
- Examples: run `python example/run_track_mopso.py` (example scripts demonstrate `ElementWiseObjective` + `MOPSO`).
- Tests live under `tests/` and follow `test_*.py` naming. Run with your preferred test runner (e.g. `pytest`) after installing dev deps.

Small implementation conventions to follow
- Prefer using the supplied helper singletons (`Logger`, `Randomizer`, `FileManager`) rather than ad-hoc logging/IO.
- When adding objectives, preserve both batch and element-wise signatures; reuse `ElementWiseObjective` when possible.
- When changing algorithm parameters, update example(s) in `example/` and add a focused test under `tests/` showing expected behavior.

Examples to copy/paste
- Constructing MOPSO (ElementWiseObjective):

```py
objective = patatune.ElementWiseObjective(my_obj, num_objectives)
mopso = patatune.MOPSO(objective=objective, lower_bounds=lb, upper_bounds=ub, num_particles=50)
pareto = mopso.optimize(num_iterations=100)
```

Files to inspect first for any change: `src/patatune/mopso/mopso.py`, `src/patatune/mopso/particle.py`, `src/patatune/objective.py`, `src/patatune/util.py`, and `tests/`.

If you need anything unclear (missing docs, expected CI commands, or dependency list mismatches like `scipy` usage vs `pyproject.toml`), ask the maintainer — I can update the instructions accordingly.

— end of short repo-specific guide
