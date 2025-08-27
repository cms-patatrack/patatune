import numpy as np
from .util import njit, get_dominated, Logger

# Generational distance
def generational_distance(pareto_front, reference_front):
    """
    This function calculates the generational distance metric, for any dimension of the pareto front.
    Parameters:
    pareto_front : numpy array
        Represents the pareto front obtained from the optimization algorithm.
    reference_front : numpy array
        Represents the true pareto front.
    Returns:
    generational_distance : float
        The generational distance metric value.
    """
    return np.mean(np.min(np.linalg.norm(pareto_front - reference_front, axis=1), axis=0))

# Inverted generational distance
def inverted_generational_distance(pareto_front, reference_front):
    """
    This function calculates the inverted generational distance metric, for any dimension of the pareto front.
    Parameters:
    pareto_front : numpy array
        Represents the pareto front obtained from the optimization algorithm.
    reference_front : numpy array
        Represents the true pareto front.
    Returns:
    inverted_generational_distance : float
        The inverted generational distance metric value.
    """
    return np.mean(np.min(np.linalg.norm(reference_front - pareto_front, axis=1), axis=0))

# Hypervolume
def hypervolume_indicator(pareto_front, reference_point, reference_hv=1, max_evaluations=10000000):
    """
    This function calculates the hypervolume indicator metric, for any dimension of the pareto front.
    Parameters:
    pareto_front : numpy array
        Represents the pareto front obtained from the optimization algorithm.
    reference_point : numpy array
        Represents the reference point for the hypervolume calculation.
    max_evaluations : int
        Maximum number of function evaluations to prevent infinite loops.
    Returns:
    hypervolume : float
        The hypervolume indicator metric value.
    """
    counter = [0] 
    result = wfg(sorted(pareto_front, key=lambda x: x[0]), reference_point, counter, max_evaluations)
    
    if counter[0] >= max_evaluations:
        Logger.warning(f"Hypervolume calculation stopped after {max_evaluations} evaluations.")
        return result/reference_hv

    return result/reference_hv


@njit
def wfg(pareto_front, reference_point, counter, max_evaluations):
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
    """
    The exclusive hypervolume of a point p relative to an underlying set S
    is the size of the part of objective space that is dominated by p but is 
    not dominated by any member of S
    """
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
    """
    return the nondominated solutions from a set of points
    """

    if len(front) == 1:
        return front
    else:
        return front[np.invert(get_dominated(front, 0))]
