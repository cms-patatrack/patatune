from copy import copy
import itertools
import math
import numpy as np
import math
from optimizer import Optimizer, FileManager, Randomizer, Logger
import scipy.stats as stats
from .particle import Particle
from optimizer.util import get_dominated


class MOPSO(Optimizer):
    """
    Multi-Objective Particle Swarm Optimization (MOPSO) algorithm.

    Parameters:
    - objective (Objective): The objective function to be optimized.
    - lower_bounds (list): The lower bounds for each parameter.
    - upper_bounds (list): The upper bounds for each parameter.
    - param_names (list): The names of the parameters (default: None).
    - num_particles (int): The number of particles in the swarm (default: 50).
    - inertia_weight (float): The inertia weight for particle velocity update (default: 0.5).
    - cognitive_coefficient (float): The cognitive coefficient for particle velocity update (default: 1).
    - social_coefficient (float): The social coefficient for particle velocity update (default: 1).
    - initial_particles_position (str): The method for initializing particle positions (default: 'random').
        Valid options are: 'lower_bounds', 'upper_bounds', 'random', 'gaussian'.
    - default_point (list): The default point for initializing particles using Gaussian distribution (default: None).
    - exploring_particles (bool): Whether to enable particle exploration (default: False).
    - topology (str): The topology of the swarm (default: 'random').
        Valid options are: 'random', 'lower_weighted_crowding_distance', 'higher_weighted_crowding_distance', 'round_robin'.
    - max_pareto_length (int): The maximum length of the Pareto front (default: -1, unlimited).

    Methods:
    - optimize(num_iterations=100, max_iterations_without_improvement=None): Runs the MOPSO optimization algorithm for a specified number of iterations.
    - step(max_iterations_without_improvement=None): Performs a single iteration of the MOPSO algorithm.
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
            # See scipy truncnorm rvs documentation for more information
            a_trunc = np.array(self.lower_bounds)
            b_trunc = np.array(self.upper_bounds)
            loc = default_point
            scale = (np.array(self.upper_bounds) -
                     np.array(self.lower_bounds))/4
            a, b = (a_trunc - loc) / scale, (b_trunc - loc) / scale
            [particle.set_position(stats.truncnorm.rvs(a, b, loc, scale))
             for particle in self.particles]

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
        Logger.debug("Saving MOPSO state")
        FileManager.save_pickle(self, "checkpoint/mopso.pkl")

    def export_state(self):
        Logger.debug("Exporting MOPSO state")
        FileManager.save_csv([np.concatenate([particle.position,
                                              particle.velocity])
                             for particle in self.particles],
                             'checkpoint/individual_states.csv',
                             headers=self.param_names + self.objective.objective_names)

        FileManager.save_csv([np.concatenate([particle.position, np.ravel(particle.fitness)])
                             for particle in self.pareto_front],
                             'checkpoint/pareto_front.csv',
                             headers=self.param_names + self.objective.objective_names)

    def load_state(self):
        Logger.debug("Loading checkpoint")
        obj = FileManager.load_pickle("checkpoint/mopso.pkl")
        self.__dict__ = obj.__dict__

    def step(self, max_iterations_without_improvement=None):
        Logger.debug(f"Iteration {self.iteration}")
        optimization_output = self.objective.evaluate(
            [particle.position for particle in self.particles])
        [particle.set_fitness(optimization_output[p_id])
            for p_id, particle in enumerate(self.particles)]
        FileManager.save_csv([np.concatenate([particle.position, np.ravel(
            particle.fitness)]) for particle in self.particles],
            'history/iteration' + str(self.iteration) + '.csv',
            headers=self.param_names + self.objective.objective_names)
        self.history[self.iteration] = np.array(
            [np.concatenate([[particle.id], particle.position, particle.velocity])
             for particle in self.particles],
            dtype=np.dtype([('id', int), ('position', float, (self.num_params,)), ('velocity', float, (self.num_params,))])
        )
        crowding_distances = self.update_pareto_front()
        self.history['pareto_front'] = np.array(
            [np.concatenate([particle.position, particle.velocity])
             for particle in self.pareto_front],
            dtype=[('position', float, (self.num_params,)), ('velocity', float, (self.num_params,))]
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
        Logger.info(f"Starting MOPSO optimization from iteration {self.iteration} to {num_iterations}")
        for _ in range(self.iteration, num_iterations):
            self.step(max_iterations_without_improvement)
            self.save_state()
            self.export_state()
        FileManager.save_zarr(self.history, 'checkpoint/history.zip', param_names=self.param_names, objective_names=self.objective.objective_names)
        return self.pareto_front

    def update_pareto_front(self):
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
        if self.objective.true_pareto is None and metric.__name__ not in ['hypervolume_indicator']:
            raise ValueError(
                "True pareto function is not defined for this objective. Only hypervolume indicators can be used.")
        pareto = np.array([particle.fitness for particle in self.pareto_front])
        x = np.linspace(0, 1, max(100, len(pareto)))
        if metric.__name__ in ['hypervolume_indicator']:
            reference_point = np.ones(len(pareto[0]))
            if self.objective.true_pareto is not None:
                reference_pareto = np.array(self.objective.true_pareto(x)).T
                reference_hypervolume = metric(reference_pareto, reference_point)
            else:
                reference_hypervolume = 1
            return metric(pareto, reference_point, reference_hypervolume)
        else:
            reference_pareto = np.array(self.objective.true_pareto(x)).T
            return metric(pareto, reference_pareto)