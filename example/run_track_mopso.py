from patatune import MOPSO
import subprocess
import os
import numpy as np
import uproot

# define the lower and upper bounds
lb = [0.0, 0.0, 0.0, 0.0] #!!!
ub = [0.006, 0.03, 0.2, 1.0] #!!!

def get_metrics(uproot_file, size):
  output = []
  for id in range(size):
    tree = uproot_file['simpleValidation' + str(id)]['output']
    total_rec = tree['rt'].array()[0]
    total_ass = tree['at'].array()[0]
    total_ass_sim = tree['ast'].array()[0]
    total_dup = tree['dt'].array()[0]
    total_sim = tree['st'].array()[0]
    
    if not total_ass or not total_rec or not total_sim or not total_ass_sim:
      output.append([1.0, 1.0])
    else:
      output.append([1 - total_ass_sim / total_sim, (total_rec - total_ass + total_dup) / total_rec])
  return output

def write_csv(filename, matrix):
    np.savetxt(filename, matrix, delimiter=',')

def reconstrucion(particles):
  write_csv('parameters.csv', [position for position in particles])
  subprocess.run(["cmsRun","../CA-parameter-tuning/reconstruction.py", "inputFiles=file:../CA-parameter-tuning/step2.root", 
                  "parametersFile=parameters.csv", "outputFile=output.root"])
  uproot_file = uproot.open('output.root')
  return get_metrics(uproot_file, len(particles))
  
  
# create the PSO object
pso = MOPSO(objective_functions=[reconstrucion],lower_bounds=lb, upper_bounds=ub, num_particles=200, num_iterations=5, inertia_weight=0.5, 
          social_coefficient=1, cognitive_coefficient=1, max_iter_no_improv=None, optimization_mode='global')

# run the optimization algorithm
pso.optimize()
