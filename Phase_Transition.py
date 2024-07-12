'''
THIS IS THE SCRIPT FOR IMPLEMENTATION OF MELTING PROBELM IN 2D SOLIDS.
|
|
Author: Tasneem Basra Khan
Date : 12 July 2024

steps followed in the implementation:
1. Choose a initial configuration
2. Write hamiltonian for energy
3. Monte carlo implementation using metropolis algorithm
4. Change the temperature for moving the particles (this will be simulated annealing)
5. plot the final configurations'''


import numpy as np
from numba import njit
import matplotlib.pyplot as plt

@njit
def hamiltonian(x, y, config, num_particles):   
    energy = 0
    for i in range(num_particles):
        energy += x[i]**2 + y[i]**2
        for j in range(num_particles):
            if j != i:
                energy += 1 / np.sqrt((x[i] - config[0, j])**2 + (y[i] - config[1, j])**2)
    return energy 

@njit
def metropolis(config, temp, num_particles):
    for i in range(num_particles):
        random = np.random.randint(0, num_particles)
        x, y = config[:, random]
        dx = x + np.random.uniform(-0.05, 0.05)
        dy = y + np.random.uniform(-0.05, 0.05)
        delta_h = (dx**2 + dy**2) - (x**2 + y**2)
        
        for j in range(num_particles):
            if j != random:
                delta_h += 1 / np.sqrt((dx - config[0, j])**2 + (dy - config[1, j])**2) - 1 / np.sqrt((x - config[0, j])**2 + (y - config[1, j])**2)
        
        prob = np.exp(-delta_h / temp)
        accept = min(1, prob)
        
        if delta_h <= 0 or accept > np.random.rand():
            config[:, random] = [dx, dy]
        else:
            config[:, random] = [x, y]
    
    return config

num_particles = 10
coordinate = np.random.rand(2, num_particles) 
config = coordinate.copy()

num_iteration = 1000
desired_temp = 0.003

for i in range(num_iteration):
    config = metropolis(config, desired_temp, num_particles)

trajectories = np.zeros((num_particles, 2, num_iteration))

for i in range(num_particles):
    current_config = config.copy()
    for j in range(num_iteration):
        current_config = metropolis(current_config, desired_temp, num_particles)
        trajectories[i, :, j] = current_config[:, i]


plt.figure(figsize=(8,8))
for i in range(num_particles):
    plt.plot(trajectories[i, 0], trajectories[i, 1] )
plt.title(f'Electron Trajectories at T = {desired_temp}'  )
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.show()
