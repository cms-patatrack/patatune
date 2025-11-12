"""Multi-Objective Particle Swarm Optimization (MOPSO) algorithm implementation."""

from copy import copy
import numpy as np
from patatune import Optimizer, FileManager, Randomizer, Logger
from .particle import Particle
from patatune.util import get_dominated


def _truncated_normal_sample(lower, upper, loc, scale_factor, max_attempts=1000):
    """Sample a truncated normal for each dimension using numpy.

    Args:
        lower (list): per-dimension lower bounds
        upper (list): per-dimension upper bounds
        loc (list): per-dimension means
        scale_factor (float): multiplicative factor for standard deviation
        max_attempts: maximum rejection sampling attempts before clipping

    Returns:
        np.ndarray: samples with same shape as loc
    """
    lower = np.array(lower, dtype=float)
    upper = np.array(upper, dtype=float)
    loc = np.array(loc, dtype=float)
    scale = np.array((upper - lower) * scale_factor, dtype=float)
    samples = np.empty_like(loc, dtype=float)
    for i in range(len(loc)):
        # Degenerate case: zero scale or identical bounds -> use clipped loc
        if scale[i] == 0 or lower[i] == upper[i]:
            samples[i] = float(np.clip(loc[i], lower[i], upper[i]))
            continue
        attempt = 0
        val = Randomizer.rng.normal(loc[i], scale[i])
        while (val < lower[i] or val > upper[i]) and attempt < max_attempts:
            val = Randomizer.rng.normal(loc[i], scale[i])
            attempt += 1
        if val < lower[i] or val > upper[i]:
            # Fallback to clipping to guarantee a value
            val = float(np.clip(val, lower[i], upper[i]))
        samples[i] = val
    return samples


class MOPSO(Optimizer):
    """ Multi-Objective Particle Swarm Optimization (MOPSO) algorithm.  
    
    The MOPSO class implements the MOPSO algorithm for multi-objective optimization problems.
    It inherits from the base Optimizer class and provides methods for initializing particles,
    updating their positions and velocities, evaluating their fitness, and maintaining the Pareto front.

    Attributes:
        objective (Objective): The functions to optimize.
        lower_bounds (list): List of lower bounds for each parameter.
        upper_bounds (list): List of upper bounds for each parameter.
            lower and upper bounds are used to check the type of each parameter (int, float, bool)
        param_names (list): List of parameter names.
        num_particles (int): Number of particles in the swarm.
        inertia_weight (float): Inertia weight for velocity update.
        cognitive_coefficient (float): Cognitive coefficient for velocity update.
        social_coefficient (float): Social coefficient for velocity update.
        initial_particles_position (str): Method for initializing particle positions. Options are `lower_bounds`, `upper_bounds`, `random`, `gaussian`.

            - if `lower_bounds`, all particles are initialized at the lower bounds;

            - if `upper_bounds`, all particles are initialized at the upper bounds;

            - if `random`, particles are initialized randomly within the bounds;

            - if `gaussian`, particles are initialized using a truncated Gaussian distribution centered around default_point.

        default_point (list): Default point for `gaussian` initialization.

            - if None, the center between lower and upper bounds is used.
        exploring_particles (bool): If True, particles that do not improve for a certain number of iterations are scattered.
        topology (str): Topology for social interaction among particles. Options are `random`, `lower_weighted_crowding_distance`, `higher_weighted_crowding_distance`, `round_robin`.

            See [Particle.get_pareto_leader][patatune.mopso.particle.Particle.get_pareto_leader] for more information.
        max_pareto_length (int): Maximum length of the Pareto front. If -1, no limit is applied.
    """
    def __init__(self,
                 objective,
                 lower_bounds, upper_bounds, param_names=None,
                 num_particles=50,
                 inertia_weight=0.5, cognitive_coefficient=1, social_coefficient=1,
                 initial_particles_position='random', default_point=None,
                 exploring_particles=False, topology='random',
                 max_pareto_length=-1):
        self.objective = objective
        self.num_particles = num_particles
        self.particles = []
        self.iteration = 0
        self.pareto_front = []
        self.max_pareto_length = max_pareto_length
        if FileManager.loading_enabled:
            try:
                self.load_state()
                return
            except FileNotFoundError as e:
                Logger.warning(
                    "Checkpoint not found. Fallback to standard construction.")
        else:
            Logger.debug("Loading disabled. Starting standard construction.")

        if len(lower_bounds) != len(upper_bounds):
            Logger.warning(f"Warning: lower_bounds and upper_bounds have different lengths."
                           f"The lowest length ({min(len(lower_bounds), len(upper_bounds))}) is taken.")
        self.num_params = min(len(lower_bounds), len(upper_bounds))
        if param_names is None:
            self.param_names = [f"param_{i}" for i in range(self.num_params)]
        else:
            if len(param_names) != self.num_params:
                raise ValueError(
                    f"Number of parameter names ({len(param_names)}) does not match number of parameters ({self.num_params}).")
            self.param_names = param_names
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        self.check_types()

        self.inertia_weight = inertia_weight
        self.cognitive_coefficient = cognitive_coefficient
        self.social_coefficient = social_coefficient
        self.particles = [Particle(lower_bounds, objective.num_objectives, num_particles, id, topology)
                          for id in range(num_particles)]

        self.exploring_particles = exploring_particles
        VALID_INITIAL_PARTICLES_POSITIONS = {
            'lower_bounds', 'upper_bounds', 'random', 'gaussian'}

        VALID_TOPOLOGIES = {
            'random', 'lower_weighted_crowding_distance', 'higher_weighted_crowding_distance', 'round_robin'}

        if topology not in VALID_TOPOLOGIES:
            raise ValueError(
                f"MOPSO: topology must be one of {VALID_TOPOLOGIES}")

        Logger.debug(f"Setting initial particles position")

        if initial_particles_position == 'lower_bounds':
            [particle.set_position(self.lower_bounds)
             for particle in self.particles]
        elif initial_particles_position == 'upper_bounds':
            [particle.set_position(self.upper_bounds)
             for particle in self.particles]
        elif initial_particles_position == 'random':
            def random_position():
                positions = []
                for i in range(self.num_params):
                    if type(self.lower_bounds[i]) == int:
                        position = Randomizer.rng.integers(
                            self.lower_bounds[i], self.upper_bounds[i])
                    elif type(self.lower_bounds[i]) == float:
                        position = Randomizer.rng.uniform(
                            self.lower_bounds[i], self.upper_bounds[i])
                    elif type(self.lower_bounds[i]) == bool:
                        position = Randomizer.rng.choice([True, False])
                    else:
                        raise ValueError(
                            f"Type {type(self.lower_bounds[i])} not supported")
                    positions.append(position)
                return np.array(positions, dtype=object)
            [particle.set_position(random_position())
             for particle in self.particles]
        elif initial_particles_position == 'gaussian':
            if default_point is None:
                default_point = np.mean(
                    [self.lower_bounds, self.upper_bounds], axis=0)
            else:
                default_point = np.array(default_point)
            for particle in self.particles:
                particle.set_position(_truncated_normal_sample(self.lower_bounds, self.upper_bounds, default_point, 0.25))

            for particle in self.particles:
                for i in range(self.num_params):
                    if type(lower_bounds[i]) == int or type(lower_bounds[i]) == bool:
                        particle.position[i] = int(round(particle.position[i]))
        elif initial_particles_position not in VALID_INITIAL_PARTICLES_POSITIONS:
            raise ValueError(
                f"MOPSO: initial_particles_position must be one of {VALID_INITIAL_PARTICLES_POSITIONS}")

        if default_point is not None:
            self.particles[0].set_position(default_point)
        self.history = {}

    def check_types(self):
        """Check that lower_bounds and upper_bounds have acceptable types and are consistent.

        Raises:
            ValueError: If any lower or upper bound has an unacceptable type,
                        or if lower_bounds and upper_bounds have inconsistent types.
        """
        lb_types = [type(lb) for lb in self.lower_bounds]
        ub_types = [type(ub) for ub in self.upper_bounds]

        acceptable_types = [int, float, bool]

        for i in range(self.num_params):
            if lb_types[i] not in acceptable_types:
                raise ValueError(f"Lower bound type {lb_types[i]} for "
                                 f"Lower bound {i} is not acceptable")
            if ub_types[i] not in acceptable_types:
                raise ValueError(f"Upper bound type {ub_types[i]} for "
                                 f"Upper bound {i} is not acceptable")

        if lb_types != ub_types:
            Logger.warning(
                "lower_bounds and upper_bounds are of different types")
            Logger.warning("Keeping the least restrictive type")
            for i in range(self.num_params):
                if lb_types[i] == float or ub_types[i] == float:
                    self.lower_bounds[i] = float(self.lower_bounds[i])
                    self.upper_bounds[i] = float(self.upper_bounds[i])
                elif lb_types[i] == int or ub_types[i] == int:
                    self.upper_bounds[i] = int(self.upper_bounds[i])
                    self.lower_bounds[i] = int(self.lower_bounds[i])

    def save_state(self):
        """Saves the current state of the MOPSO optimizer to a checkpoint file.
        
        Uses the FileManager to serialize and save the MOPSO object to 'checkpoint/mopso.pkl'.
        """
        Logger.debug("Saving MOPSO state")
        FileManager.save_pickle(self, "checkpoint/mopso.pkl")

    def export_state(self):
        """Exports the current state of the MOPSO optimizer to CSV files.

        Uses the FileManager to export:
         - the states of individual particles to 'checkpoint/individual_states.csv'
         - the current Pareto front to 'checkpoint/pareto_front.csv'.
        """
        Logger.debug("Exporting MOPSO state")
        FileManager.save_csv([np.concatenate([particle.position,
                                              particle.velocity])
                             for particle in self.particles],
                             'checkpoint/individual_states.csv',
                             headers=self.param_names + [f"velocity_{p}" for p in self.param_names])

        FileManager.save_csv([np.concatenate([particle.position, np.ravel(particle.fitness * self.objective.directions)])
                             for particle in self.pareto_front],
                             'checkpoint/pareto_front.csv',
                             headers=self.param_names + self.objective.objective_names)

    def load_state(self):
        """Loads the MOPSO optimizer state from a checkpoint file.
        
        Uses the FileManager to deserialize and restore the MOPSO object from 'checkpoint/mopso.pkl'.
        """
        Logger.debug("Loading checkpoint")
        obj = FileManager.load_pickle("checkpoint/mopso.pkl")
        self.__dict__ = obj.__dict__

    def step(self, max_iterations_without_improvement=None):
        """Performs a single optimization step in the MOPSO algorithm.
        
        Args:
            max_iterations_without_improvement (int, optional): Maximum number of iterations a particle can go
                without improvement before being scattered. If None, no scattering is performed.
        """
        Logger.debug(f"Iteration {self.iteration}")
        optimization_output = self.objective.evaluate(
            [particle.position for particle in self.particles])
        [particle.set_fitness(optimization_output[p_id])
            for p_id, particle in enumerate(self.particles)]
        FileManager.save_csv([np.concatenate([particle.position, np.ravel(
            particle.fitness * self.objective.directions)]) for particle in self.particles],
            'history/iteration' + str(self.iteration) + '.csv',
            headers=self.param_names + self.objective.objective_names)
        self.history[self.iteration] = np.array(
            [(particle.id, particle.position, particle.fitness * self.objective.directions) for particle in self.particles],
            dtype=np.dtype([('id', int), ('position', float, (self.num_params,)), ('fitness', float, (self.objective.num_objectives,))])
        )
        crowding_distances = self.update_pareto_front()
        self.history['pareto_front'] = np.array(
            [(particle.position, particle.fitness * self.objective.directions) for particle in self.pareto_front],
            dtype=[('position', float, (self.num_params,)), ('fitness', float, (self.objective.num_objectives,))]
        )
        for particle in self.particles:
            particle.update_velocity(self.pareto_front,
                                     crowding_distances,
                                     self.inertia_weight,
                                     self.cognitive_coefficient,
                                     self.social_coefficient)
            if self.exploring_particles and max_iterations_without_improvement and particle.iterations_with_no_improvement >= max_iterations_without_improvement:
                self.scatter_particle(particle)
            particle.update_position(self.lower_bounds, self.upper_bounds)
        self.iteration += 1

    def optimize(self, num_iterations=100, max_iterations_without_improvement=None):
        """Runs the MOPSO optimization process for a specified number of iterations.
        
        Uses the `step` method to perform optimization steps and manages the overall optimization loop.

        Args:
            num_iterations (int): Total number of iterations to perform.
            max_iterations_without_improvement (int, optional): Maximum number of iterations a particle can go
                without improvement before being scattered. If None, no scattering is performed.

        Returns:
            (list): The final Pareto front after optimization.
        """
        Logger.info(f"Starting MOPSO optimization from iteration {self.iteration} to {num_iterations}")
        for _ in range(self.iteration, num_iterations):
            self.step(max_iterations_without_improvement)
            self.save_state()
            self.export_state()
        FileManager.save_zarr(self.history, 'checkpoint/mopso.zip', param_names=self.param_names, objective_names=self.objective.objective_names, lower_bounds=self.lower_bounds, upper_bounds=self.upper_bounds)
        return self.pareto_front

    def update_pareto_front(self):
        """Updates the Pareto front based on the current particles' fitness.
        
        Returns:
            (dict): A dictionary mapping each particle in the Pareto front to its crowding distance.
        """
        Logger.debug("Updating Pareto front")
        pareto_lenght = len(self.pareto_front)
        particles = self.pareto_front + self.particles
        particle_fitnesses = np.array(
            [particle.fitness for particle in particles])
        dominanted = get_dominated(particle_fitnesses, pareto_lenght)

        self.pareto_front = [copy(particles[i]) for i in range(
            len(particles)) if not dominanted[i]]
        crowding_distances = self.calculate_crowding_distance(
            self.pareto_front)
        self.pareto_front.sort(
            key=lambda x: crowding_distances[x], reverse=True)

        if self.max_pareto_length > 0:
            self.pareto_front = self.pareto_front[: self.max_pareto_length]
            
        Logger.debug(f"New pareto front size: {len(self.pareto_front)}")

        crowding_distances = self.calculate_crowding_distance(
            self.pareto_front)
        return crowding_distances

    def calculate_crowding_distance(self, pareto_front):
        """Calculates the crowding distance for each particle in the Pareto front.

        Args:
            pareto_front (list): List of particles representing the current Pareto front.
        
        Returns:
            (dict): A dictionary mapping each particle in the Pareto front to its crowding distance.
        """
        if len(pareto_front) == 0:
            return []
        num_objectives = len(np.ravel(pareto_front[0].fitness))
        distances = [0] * len(pareto_front)
        point_to_distance = {}
        for i in range(num_objectives):
            # Sort by objective i
            sorted_front = sorted(
                pareto_front, key=lambda x: np.ravel(x.fitness)[i])
            # Set the boundary points to infinity
            distances[0] = float('inf')
            distances[-1] = float('inf')
            # Normalize the objective values for calculation
            min_obj = np.ravel(sorted_front[0].fitness)[i]
            max_obj = np.ravel(sorted_front[-1].fitness)[i]
            norm_denom = max_obj - min_obj if max_obj != min_obj else 1
            for j in range(1, len(pareto_front) - 1):
                distances[j] += (np.ravel(sorted_front[j + 1].fitness)[i] -
                                 np.ravel(sorted_front[j - 1].fitness)[i]) / norm_denom
        for i, point in enumerate(pareto_front):
            point_to_distance[point] = distances[i]
        return point_to_distance

    def scatter_particle(self, particle: Particle):
        """Scatters a particle that has not improved for a certain number of iterations.

        The particle's velocity is adjusted to move it towards less crowded areas of the search space.

        Args:
            particle (Particle): The particle to be scattered.
        """
        Logger.debug(
            f"Particle {particle} did not improve for 10 iterations. Scattering.")
        for i in range(len(self.lower_bounds)):
            lower_count = sum(
                [1 for p in self.particles if p.position[i] < particle.position[i]])
            upper_count = sum(
                [1 for p in self.particles if p.position[i] > particle.position[i]])
            if lower_count > upper_count:
                particle.velocity[i] = 1
            else:
                particle.velocity[i] = -1

    def get_metric(self, metric):
        """Calculates a specified metric for the current Pareto front.

        For example:
        ``` python
        mopso.get_metric(patatune.metrics.generational_distance)
        ```

        Args:
            metric (function): A [metric][patatune.metrics] function that takes two arguments: the Pareto front and the reference front.
        
        Returns:
            (float): The calculated metric value.
        """
        result = None
        if self.objective.true_pareto is None and metric.__name__ not in ['hypervolume_indicator']:
            raise ValueError(
                "True pareto function is not defined for this objective. Only hypervolume indicators can be used.")
        pareto = np.array([particle.fitness for particle in self.pareto_front])
        if metric.__name__ in ['hypervolume_indicator']:
            reference_point = np.ones(len(pareto[0]))
            if self.objective.true_pareto is not None:
                reference_pareto = self.objective.true_pareto(len(pareto))
                reference_hypervolume = metric(reference_pareto, reference_point)
            else:
                reference_hypervolume = 1
            Logger.debug(f"Measuring {metric.__name__}. Reference hypervolume: {reference_hypervolume}")
            result = metric(pareto, reference_point, reference_hypervolume)
        else:
            reference_pareto = self.objective.true_pareto(len(pareto))
            Logger.debug(f"Measuring {metric.__name__}")
            result = metric(pareto, reference_pareto)
        Logger.info(f"{metric.__name__}: {result}")
        return result