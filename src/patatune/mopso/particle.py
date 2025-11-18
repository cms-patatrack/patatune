"""Module defining the Particle class for the MOPSO algorithm."""

import numpy as np
from patatune import Randomizer
from patatune.util import get_dominated


class Particle:
    """Class representing a particle in the MOPSO algorithm.

    Attributes:
        position (np.ndarray): Current position of the particle in the search space.
        num_objectives (int): Number of objectives in the optimization problem.
        num_particles (int): Total number of particles in the swarm.
        velocity (np.ndarray): Current velocity of the particle.
        fitness (np.ndarray): Current fitness values of the particle for each objective.
        local_best_fitnesses (list): List of local best fitness values found by the particle.
        local_best_positions (list): List of positions corresponding to the local best fitnesses.
        iterations_with_no_improvement (int): Counter for iterations without improvement.
        id (int): Unique identifier for the particle.
        topology (str): Topology strategy for selecting global best in the swarm.
    """
    def __init__(self, lower_bound, num_objectives, num_particles, id, topology):
        self.position = np.asarray(lower_bound)
        self.num_objectives = num_objectives
        self.num_particles = num_particles
        self.velocity = np.zeros_like(self.position)

        self.fitness = np.full(self.num_objectives, np.inf)
        self.local_best_fitnesses = []
        self.local_best_positions = []
        self.iterations_with_no_improvement = 0
        self.id = id
        self.topology = topology

    def update_velocity(self,
                        pareto_front,
                        crowding_distances,
                        inertia_weight=0.5,
                        cognitive_coefficient=1,
                        social_coefficient=1):
        """Updates the velocity of the particle based on its local best and the global best.

        Args:
            pareto_front (list): List of particles representing the current Pareto front.
            crowding_distances (dict): Dictionary mapping particles to their crowding distances.
            inertia_weight (float): Weight for the inertia component (default: 0.5).
            cognitive_coefficient (float): Coefficient for the cognitive component (default: 1).
            social_coefficient (float): Coefficient for the social component (default: 1).
        """
        leader = self.get_pareto_leader(pareto_front, crowding_distances)
        best_position = Randomizer.rng.choice(self.local_best_positions)
        cognitive_random = Randomizer.rng.uniform(0, 1)
        social_random = Randomizer.rng.uniform(0, 1)

        def cast_position(arr):
            # Convert boolean and integer types to float for velocity calculation
            return np.array([int(x) if isinstance(x, bool) else x for x in arr], dtype=float)

        best_position_cast = cast_position(best_position)
        position_cast = cast_position(self.position)
        leader_position_cast = cast_position(leader.position)

        cognitive = cognitive_coefficient * cognitive_random * (best_position_cast - position_cast)
        social = social_coefficient * social_random * (leader_position_cast - position_cast)
        self.velocity = inertia_weight * self.velocity + cognitive + social

    def update_position(self, lower_bound, upper_bound):
        """Updates the position of the particle based on its velocity and the problem boundaries.

        If the variable is of integer type, the new position is rounded.
        If the variable is of boolean type, the new position is determined by a threshold of 0.5.

        Args:
            lower_bound (np.ndarray): Lower bounds for each dimension of the search space.
            upper_bound (np.ndarray): Upper bounds for each dimension of the search space.
        """
        new_position = np.empty_like(self.position)
        for i in range(len(lower_bound)):
            if type(lower_bound[i]) == int:
                new_position[i] = np.round(self.position[i] + self.velocity[i])
            elif type(lower_bound[i]) == bool:
                new_position[i] = self.position[i] + self.velocity[i] > 0.5
            else:
                new_position[i] = self.position[i] + self.velocity[i]
        self.position = np.clip(new_position, lower_bound, upper_bound)

    def set_fitness(self, fitness):
        """Sets the fitness of the particle and updates its local best if necessary.

        Args:
            fitness (np.ndarray): New fitness values for the particle.
        """
        self.fitness = fitness
        self.update_best()

    def set_position(self, position):
        """Sets the position of the particle.

        Args:
            position (np.ndarray): New position for the particle.
        """
        self.position = position

    def set_state(self, velocity, position, best_position, fitness, best_fitness):
        """Sets the complete state of the particle.

        Args:
            velocity (np.ndarray): New velocity for the particle.
            position (np.ndarray): New position for the particle.
            best_position (list): New list of local best positions for the particle.
            fitness (np.ndarray): New fitness values for the particle.
            best_fitness (list): New list of local best fitness values for the particle.
        """
        self.velocity = velocity
        self.position = position
        self.best_position = best_position
        self.fitness = fitness
        self.best_fitness = best_fitness

    def update_best(self):
        """Updates the local best fitnesses and positions of the particle based on its current fitness.
        
        Uses the get_dominated utility function to identify non-dominated solutions.
        Resets the iterations_with_no_improvement counter if there is an improvement.
        """
        len_fitness = len(self.local_best_fitnesses)
        fitnesses = np.array(
            [f for f in self.local_best_fitnesses] + [self.fitness])
        positions = np.array(
            [p for p in self.local_best_positions] + [self.position])

        dominated = get_dominated(fitnesses, len_fitness)

        new_local_best_fitnesses = [fitnesses[i]
                                    for i in range(len(fitnesses)) if not dominated[i]]

        # if the new fitness is different from the old one, reset the counter
        if np.array_equal(new_local_best_fitnesses, self.local_best_fitnesses):
            self.iterations_with_no_improvement += 1
        else:
            self.iterations_with_no_improvement = 0
        self.local_best_fitnesses = new_local_best_fitnesses
        self.local_best_positions = [positions[i]
                                     for i in range(len(positions)) if not dominated[i]]

    def get_pareto_leader(self, pareto_front, crowding_distances):
        """Selects a leader particle from the Pareto front based on the specified topology.

        If the topology is "random", a random particle from the Pareto front is selected.
        If the topology is "lower_weighted_crowding_distance", a particle is selected
        with a probability inversely proportional to its crowding distance calling the [weighted_crowding_distance_topology][patatune.mopso.particle.weighted_crowding_distance_topology] function.
        If the topology is "higher_weighted_crowding_distance", a particle is selected
        with a probability proportional to its crowding distance calling the [weighted_crowding_distance_topology][patatune.mopso.particle.weighted_crowding_distance_topology] function.
        If the topology is "round_robin", particles are selected in a round-robin fashion
        based on the particle's ID calling the [round_robin_topology][patatune.mopso.particle.round_robin_topology] function.

        Args:
            pareto_front (list): List of particles representing the current Pareto front.
            crowding_distances (dict): Dictionary mapping particles to their crowding distances.
        """
        if self.topology == "random":
            return Randomizer.rng.choice(pareto_front)
        elif self.topology == "lower_weighted_crowding_distance":
            return weighted_crowding_distance_topology(pareto_front, crowding_distances, higher=False)
        elif self.topology == "higher_weighted_crowding_distance":
            return weighted_crowding_distance_topology(pareto_front, crowding_distances, higher=True)
        elif self.topology == "round_robin":
            return round_robin_topology(pareto_front, self.id)
        else:
            raise ValueError(
                f"MOPSO: {self.topology} not implemented!")


def weighted_crowding_distance_topology(pareto_front, crowding_distances, higher):
    """ Selects a leader particle from the Pareto front based on crowding distances.
    
    Args:
        pareto_front (list): List of particles representing the current Pareto front.
        crowding_distances (dict): Dictionary mapping particles to their crowding distances.
        higher (bool): If True, selects particles with higher crowding distances with higher probability.
                        If False, selects particles with lower crowding distances with higher probability.
    
    Returns:
        (Particle): The selected leader particle from the Pareto front.
    """
    pdf = boltzmann(crowding_distances, higher)
    return Randomizer.rng.choice(pareto_front, p=pdf)


def round_robin_topology(pareto_front, id):
    """ Selects a leader particle from the Pareto front in a round-robin fashion.

    The particle is selected based on its ID modulo the size of the Pareto front.
    If the ID exceeds the size of the Pareto front, it wraps around.
    
    Args:
        pareto_front (list): List of particles representing the current Pareto front.
        id (int): Unique identifier for the particle.
    
    Returns:
        (Particle): The selected leader particle from the Pareto front.
    """
    index = id % len(pareto_front)
    return pareto_front[index]


def boltzmann(crowding_distances, higher):
    """ Computes a probability distribution function (PDF) based on crowding distances.

    Args:
        crowding_distances (dict): Dictionary mapping particles to their crowding distances.
        higher (bool): If True, computes PDF favoring higher crowding distances.
                        If False, computes PDF favoring lower crowding distances.
    
    Returns:
        (list): A list representing the probability distribution function for selecting particles.
    """
    cd_list = list(crowding_distances.values())
    len_cd = len(cd_list)
    if len_cd == 1:
        return [1]
    if len_cd == 2:
        return Randomizer.rng.choice([[1, 0], [0, 1]])
    pdf = np.empty(len_cd)
    if higher:
        cd_list = [1 / (cd + 1e-9) for cd in cd_list]
    for i in range(len_cd):
        pdf[i] = np.exp(-cd_list[i])
    pdf = pdf / (np.sum(pdf))
    return pdf
