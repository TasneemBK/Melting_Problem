'''THIS SCRIPT IS FOR IMPLEMENTATION OF ISING MODEL IN 2D SOLIDS.

Source of the code: https://youtu.be/K--1hlv9yv0?si=gcD7NcFL6M1jBTD1'''


#coding for ising model

import numpy as np
import matplotlib.pyplot as plt
#plt.style.use(['science' , 'notebook' , 'grid'])   
import numba
from numba import njit   
from scipy.ndimage import convolve , generate_binary_structure
import random

#make a 50 by 50 grind of spins 
N = 50  
spins = np.random.choice([-1,1], size=(N,N))    
plt.imshow(spins , cmap='binary')       
plt.show()  

matrix = [[random.choice([-1,1]) for i in range(N)]for j in range(N)] 
array = np.array(matrix, dtype=np.float64) 
print(matrix)

def get_energy(spins):
    kern = generate_binary_structure(2,1)  
    kern[1][1] = False
    arr = - array * convolve(array, kern, mode='constant' , cval=0.0)    
    return arr.sum()
#convolve is the superpositioning of kern on array
print(get_energy(array)  )

#metropolis algorithm   
@numba.njit("UniTuple(f8[:], 2)(f8[:,:], i8, f8, f8)", nopython=True , nogil=True) 
def metropolis(array , times , BJ , energy):
    spin_arr = array.copy()
    net_spin = np.zeros(times -1) #forming a list
    net_energy = np.zeros(times -1) 
    for t in range(times):
        x = np.random.randint(0, N) 
        y = np.random.randint(0, N) 
        spin_i = spin_arr[x,y]  
        spin_f = spin_i * -1    

        E_i = 0 
        E_f = 0 
        if x > 0 :
            E_i += -spin_i * spin_arr[x-1 , y] 
            E_f += -spin_f * spin_arr[x-1 , y]   
        if x < N-1: 
            E_i += -spin_i * spin_arr[x+1 , y] 
            E_f += -spin_f * spin_arr[x+1 , y]  
        if y > 0:   
            E_i += -spin_i * spin_arr[x , y-1] 
            E_f += -spin_f * spin_arr[x , y-1]  
        if y < N-1: 
            E_i += -spin_i * spin_arr[x , y+1] 
            E_f += -spin_f * spin_arr[x , y+1]  
        delta_E = E_f - E_i 
        if (delta_E > 0)*(np.random.random() < np.exp(-BJ*delta_E)):
            spin_arr[x,y] = spin_f 
            energy += delta_E   
        elif delta_E <= 0: 
            spin_arr[x,y] = spin_f 
            energy += delta_E

        net_spin[t] = spin_arr.sum()
        net_energy[t] = energy   

    return net_energy , net_spin 

spins , energies = metropolis(array , 1000000 , 0.7 , get_energy(array))  

fig , axes = plt.subplots(1,2 , figsize=(12,4)) 
ax= axes[0]
ax.plot(spins/N**2)
ax.set_xlabel('time')   
ax.set_ylabel('magnetization')  
ax.grid
ax= axes[1] 
ax.plot(energies)   
ax.set_xlabel('time')   
ax.set_ylabel('energy') 
ax.grid()   
plt.show()  
