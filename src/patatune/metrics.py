"""Module implementing various multi-objective optimization metrics."""

import numpy as np
from .util import njit, get_dominated, Logger

def generational_distance(pareto_front, reference_front):
    """Calculates the generational distance metric, for any dimension of the pareto front.

    The generational distance (GD) measures the average distance of points in the obtained Pareto front to the nearest point in the true Pareto front.

    Args:
        pareto_front (np.ndarray): Represents the pareto front obtained from the optimization algorithm.
        reference_front (np.ndarray): Represents the true pareto front.

    Returns:
        (float): The generational distance metric value.
    """
    if len(pareto_front) == 0 or len(reference_front) == 0:
        return float('inf')
    
    sum_distances = 0.0
    for p in pareto_front:
        min_distance = float('inf')
        for r in reference_front:
            distance = np.linalg.norm(p - r)
            if distance < min_distance:
                min_distance = distance
        sum_distances += min_distance ** 2
    
    return (sum_distances / len(pareto_front)) ** 0.5

def inverted_generational_distance(pareto_front, reference_front):
    """Calculates the inverted generational distance metric, for any dimension of the pareto front.

    The inverted generational distance (IGD) measures the average distance of points in the true Pareto front to the nearest point in the obtained Pareto front.

    Args:
        pareto_front (np.ndarray): Represents the pareto front obtained from the optimization algorithm.
        reference_front (np.ndarray): Represents the true pareto front.

    Returns:
        (float): The inverted generational distance metric value.
    """
    if len(reference_front) == 0 or len(pareto_front) == 0:
        return float('inf')
    
    sum_distances = 0.0
    for r in reference_front:
        min_distance = float('inf')
        for p in pareto_front:
            distance = np.linalg.norm(r - p)
            if distance < min_distance:
                min_distance = distance
        sum_distances += min_distance ** 2

    return (sum_distances / len(reference_front)) ** 0.5

def hypervolume_indicator(pareto_front, reference_point, reference_hv=1, max_evaluations=10000000):
    """Calculates the hypervolume indicator metric, for any dimension of the pareto front.

    The hypervolume indicator (HV) measures the volume of the objective space dominated by the obtained Pareto front and bounded by a reference point.

    Args:
        pareto_front (np.ndarray): Represents the pareto front obtained from the optimization algorithm.
        reference_point (list or np.ndarray): A reference point in the objective space, typically chosen to be worse than any point in the pareto front.
        reference_hv (float): The hypervolume of the reference front for normalization (default: 1).
        max_evaluations (int): Maximum number of evaluations to perform during hypervolume calculation (default: 10,000,000).
            Maximum number of evaluations to perform during hypervolume calculation (default: 10,000,000).
    
    Returns:
        (float): The hypervolume indicator metric value normalized by the reference hypervolume.
    """
    counter = [0] 
    result = wfg(sorted(pareto_front, key=lambda x: x[0]), reference_point, counter, max_evaluations)
    
    if counter[0] >= max_evaluations:
        Logger.warning(f"Hypervolume calculation stopped after {max_evaluations} evaluations.")
        return result/reference_hv

    return result/reference_hv


@njit
def wfg(pareto_front, reference_point, counter, max_evaluations):
    """
    WFG algorithm for hypervolume calculation
    Reference: While, L., Bradstreet, L., & Barone, L. (2012). A fast way of calculating exact hypervolumes.
    IEEE Transactions on Evolutionary Computation, 16(1), 86-95.
    DOI: [10.1109/TEVC.2010.2077298](https://doi.org/10.1109/TEVC.2010.2077298)

    Args:
        pareto_front (np.ndarray): Represents the pareto front obtained from the optimization algorithm.
        reference_point (list or np.ndarray): A reference point in the objective space, typically chosen to be worse than any point in the pareto front.
        counter (list): A list containing a single integer to keep track of the number of evaluations performed.
        max_evaluations (int): Maximum number of evaluations to perform during hypervolume calculation.

    Returns:
        (float): The hypervolume of the pareto front with respect to the reference point.

    Note:
        Optionally uses numba's njit for performance optimization.
    """

    if counter is None:
        counter = [0]
    
    # Don't return 0 immediately - let it compute partial results
    counter[0] += 1
    
    if len(pareto_front) == 0: 
        return 0
    else:
        sum = 0
        for k in range(len(pareto_front)):
            if counter[0] >= max_evaluations:
                # Return partial sum computed so far
                break
            sum = sum + exclhv(pareto_front, k, reference_point, counter, max_evaluations)
        return sum

@njit
def exclhv(pareto_front, k, reference_point, counter, max_evaluations):
    if counter is None:
        counter = [0]
    
    counter[0] += 1
    
    # Always compute at least the inclusive hypervolume
    result = inclhv(pareto_front[k], reference_point)
    
    # Only try to subtract if we haven't hit the limit yet
    if counter[0] < max_evaluations:
        limited_set = limitset(pareto_front, k)
        if len(limited_set) > 0:
            result = result - wfg(nds(limited_set), reference_point, counter, max_evaluations)
    
    return result

@njit
def inclhv(p, reference_point):
    volume = 1
    for i in range(len(p)):
        volume = volume * max(0, reference_point[i] - p[i])
    return volume

@njit
def limitset(pareto_front, k):
    m = len(pareto_front) - k - 1
    n = len(pareto_front[0])
    result = np.empty((m, n))
    for j in range(m):
        l = np.empty(n)
        for i in range(n):
            p = pareto_front[k][i]
            q = pareto_front[j+k+1][i]
            l[i] = p if p > q else q
        result[j] = l
    return result

@njit
def nds(front):
    """ Returns the non-dominated set from the given front. 
    
    Uses the get_dominated utility function to identify dominated points and filters them out.

    Args:
        front (np.ndarray): Represents a set of points in the objective space.
    Returns:
        (np.ndarray): The non-dominated subset of the input front.
    """

    if len(front) == 1:
        return front
    else:
        return front[np.invert(get_dominated(front, 0))]
