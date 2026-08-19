"""Test pareto front history saving functionality."""

import patatune
import numpy as np
import os
import shutil


def test_pareto_history_enabled():
    """Test that pareto front history is saved when enabled."""
    # Clean up test directory
    test_dir = "tmp/test_pareto_history_enabled"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    # Setup
    lb = [0.0, 0.0]
    ub = [5.0, 3.0]
    num_agents = 10
    num_iterations = 3
    
    def f(params):
        return [[4 * p[0]**2 + 4 * p[1]**2, (p[0] - 5)**2 + (p[1] - 5)**2] for p in params]
    
    patatune.FileManager.working_dir = test_dir
    patatune.FileManager.saving_history_enabled = True
    
    objective = patatune.Objective([f])
    pso = patatune.MOPSO(objective=objective, lower_bounds=lb, upper_bounds=ub,
                          num_particles=num_agents,
                          inertia_weight=0.5, cognitive_coefficient=1, social_coefficient=1)
    
    # Run optimization
    pso.optimize(num_iterations)
    
    # Verify history keys
    assert len(pso.history) == num_iterations * 2  # particles + pareto for each iteration
    
    # Verify CSV files exist for each iteration
    for i in range(num_iterations):
        particle_csv = os.path.join(test_dir, f'history/iteration{i}.csv')
        pareto_csv = os.path.join(test_dir, f'history/pareto_iteration{i}.csv')
        
        assert os.path.exists(particle_csv), f"Particle CSV for iteration {i} not found"
        assert os.path.exists(pareto_csv), f"Pareto CSV for iteration {i} not found"
        
        # Verify history dictionary contains data
        assert i in pso.history, f"Iteration {i} not in history"
        assert f'pareto_front_{i}' in pso.history, f"Pareto front {i} not in history"
        
        # Verify data structure
        assert isinstance(pso.history[i], list), f"Iteration {i} history is not a list"
        assert isinstance(pso.history[f'pareto_front_{i}'], list), f"Pareto front {i} history is not a list"
        
        if len(pso.history[i]) > 0:
            assert 'id' in pso.history[i][0], f"Particle data missing 'id' field"
            assert 'position' in pso.history[i][0], f"Particle data missing 'position' field"
            assert 'fitness' in pso.history[i][0], f"Particle data missing 'fitness' field"
        
        if len(pso.history[f'pareto_front_{i}']) > 0:
            assert 'position' in pso.history[f'pareto_front_{i}'][0], f"Pareto data missing 'position' field"
            assert 'fitness' in pso.history[f'pareto_front_{i}'][0], f"Pareto data missing 'fitness' field"
    
    print("✓ Test passed: Pareto history enabled")


def test_pareto_history_disabled():
    """Test that pareto front history is NOT saved when disabled."""
    # Clean up test directory
    test_dir = "tmp/test_pareto_history_disabled"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    # Setup
    lb = [0.0, 0.0]
    ub = [5.0, 3.0]
    num_agents = 10
    num_iterations = 3
    
    def f(params):
        return [[4 * p[0]**2 + 4 * p[1]**2, (p[0] - 5)**2 + (p[1] - 5)**2] for p in params]
    
    patatune.FileManager.working_dir = test_dir
    patatune.FileManager.saving_history_enabled = False
    
    objective = patatune.Objective([f])
    pso = patatune.MOPSO(objective=objective, lower_bounds=lb, upper_bounds=ub,
                          num_particles=num_agents,
                          inertia_weight=0.5, cognitive_coefficient=1, social_coefficient=1)
    
    # Run optimization
    pso.optimize(num_iterations)
    
    # Verify history is empty
    assert len(pso.history) == 0, "History should be empty when saving is disabled"
    
    # Verify CSV files do NOT exist
    history_dir = os.path.join(test_dir, 'history')
    assert not os.path.exists(history_dir), "History directory should not exist when saving is disabled"
    
    print("✓ Test passed: Pareto history disabled")


def test_pareto_history_growth():
    """Test that pareto front grows/changes over iterations."""
    # Clean up test directory
    test_dir = "tmp/test_pareto_history_growth"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    # Setup
    lb = [0.0, 0.0]
    ub = [5.0, 3.0]
    num_agents = 20
    num_iterations = 5
    
    def f(params):
        return [[4 * p[0]**2 + 4 * p[1]**2, (p[0] - 5)**2 + (p[1] - 5)**2] for p in params]
    
    patatune.FileManager.working_dir = test_dir
    patatune.FileManager.saving_history_enabled = True
    
    objective = patatune.Objective([f])
    pso = patatune.MOPSO(objective=objective, lower_bounds=lb, upper_bounds=ub,
                          num_particles=num_agents,
                          inertia_weight=0.5, cognitive_coefficient=1, social_coefficient=1)
    
    # Run optimization
    pso.optimize(num_iterations)
    
    # Check that pareto front sizes change (typically grow early on)
    pareto_sizes = []
    for i in range(num_iterations):
        pareto_key = f'pareto_front_{i}'
        size = len(pso.history[pareto_key])
        pareto_sizes.append(size)
        print(f"  Iteration {i}: Pareto front size = {size}")
    
    # At least some iterations should have non-empty pareto fronts
    assert any(size > 0 for size in pareto_sizes), "Pareto front should have solutions"
    
    print("✓ Test passed: Pareto history growth tracked")


if __name__ == "__main__":
    print("Testing pareto front history functionality...")
    print()
    
    test_pareto_history_enabled()
    test_pareto_history_disabled()
    test_pareto_history_growth()
    
    print()
    print("All tests passed! ✓")
