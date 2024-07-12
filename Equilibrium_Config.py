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

import random
import numba
import numpy as np  
from numba import njit
import matplotlib.pyplot  as plt

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
def metropolis(config, temp , num_particles):   
    for i in range(num_particles):
        random = np.random.randint(0, num_particles)
        x, y = config[:, random]
        dx = x + np.random.uniform(-1,1)
        dy = y + np.random.uniform(-1,1)
        delta_h = (dx**2 + dy**2) - (x**2 + y**2)
        
        for j in range(num_particles):
            if j != random:
                delta_h += 1/ np.sqrt((dx - config[0, j])**2 + (dy - config[1, j])**2) - 1 / np.sqrt((x - config[0, j])**2 + (y - config[1, j])**2)
        
        prob = np.exp(-delta_h / temp)
        accept = min(1, prob)
        
        if delta_h <= 0 or accept > np.random.rand():
            config[:, random] = [dx, dy]
        else : 
            config[:, random] = [x, y ]
        
    return config

num_particles = 50
coordinate = np.random.rand(2, num_particles)
x = coordinate[0]
y = coordinate[1]
config = np.array([x, y])
initial_energy = hamiltonian(x, y, config, num_particles)
print(f"Initial energy = {initial_energy}")
plt.figure(figsize=(10,10))
plt.title('Initial configuration')
plt.scatter(config[0], config[1])
plt.show()

num_iteration = 10000
temp = 0.00001

for i in range(num_iteration):
    config = metropolis(config,  temp,  num_particles)

final_energy = hamiltonian(config[0], config[1], config, num_particles)
print(f"Final energy = {final_energy}")
plt.figure(figsize=(8,8))
plt.title('Final configuration')
plt.scatter(config[0], config[1])
plt.show()
