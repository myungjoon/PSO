import numpy as np
import matplotlib.pyplot as plt
from fitness_functions import *
from pyDOE import *

path = './supportData/'
result_filename = 'fitness1_5000.txt'

prob_num = 1

####constants
w = 0.729
C1 = 2.05
C2 = 2.05

#########################
##define solution space##
#########################


d = 30
x_min = np.zeros(d)
x_max = np.zeros(d)
    
if prob_num in [1,2,3,4,5,6]:
    for i in range(d):
        x_min[i] = -100.0
        x_max[i] = 100.0
elif prob_num == 7:
    for i in range(d):
        x_min[i] = 0
        x_max[i] = 600
elif prob_num == 8:
    for i in range(d):
        x_min[i] = -32
        x_max[i] = 32    
elif prob_num in [9, 10]:
    for i in range(d):
        x_min[i] = -5
        x_max[i] = 5

d_max = x_max - x_min

# problem constants

o, M, A = None, None, None

with open(path + 'sphere_func_data.txt', 'r') as f:
    o1 = np.array(list(map(float, f.readline().strip().split())))[:d]

with open(path + 'schwefel_102_data.txt', 'r') as f:
    o2 = np.array(list(map(float,f.readline().strip().split())))[:d]
    
with open(path + 'high_cond_elliptic_rot_data.txt', 'r') as f:
    o3 = np.array(list(map(float,f.readline().strip().split())))[:d]

with open(path + 'elliptic_M_D30.txt', 'r') as f:
            M3 = np.zeros([d,d])
            for i in range(d):
                line = f.readline()
                M3[i,:] = np.array(list(map(float,line.strip().split())))[:d]
                
with open(path + 'schwefel_102_data.txt', 'r') as f:
    o4 = np.array(list(map(float,f.readline().strip().split())))[:d]
    
with open(path + 'schwefel_206_data.txt', 'r') as f:
    o5 = np.array(list(map(float,f.readline().strip().split())))[:d]
    A5 = np.zeros([d,d])
    i = 0
    for i in range(d):
        line = f.readline()
        A5[i,:] = np.array(list(map(float, line.strip().split())))[:d]
            
with open(path + 'rosenbrock_func_data.txt', 'r') as f:
    o6 = np.array(list(map(float,f.readline().strip().split())))[:d]

with open(path + 'griewank_func_data.txt', 'r') as f:
    o7 = np.array(list(map(float,f.readline().strip().split())))[:d]
    
with open(path + 'griewank_M_D30.txt', 'r') as f:
    M7 = np.zeros([d,d])
    for i in range(d):
        line = f.readline()
        M7[i,:] = np.array(list(map(float,line.strip().split())))[:d]
        
with open(path + 'ackley_func_data.txt', 'r') as f:
    o8 = np.array(list(map(float,f.readline().strip().split())))[:d]
    
with open(path + 'ackley_M_D30.txt', 'r') as f:
    M8 = np.zeros([d,d])
    for i in range(d):
        line = f.readline()
        M8[i,:] = np.array(list(map(float,line.strip().split())))[:d]
        
with open(path + 'rastrigin_func_data.txt', 'r') as f:
    o9 = np.array(list(map(float,f.readline().strip().split())))[:d]
    
with open(path + 'rastrigin_func_data.txt', 'r') as f:
    o10 = np.array(list(map(float,f.readline().strip().split())))[:d]
    
with open(path + 'rastrigin_M_D30.txt', 'r') as f:
    M10 = np.zeros([d,d])
    for i in range(d):
        line = f.readline()
        M10[i,:] = np.array(list(map(float,line.strip().split())))[:d]

if prob_num == 1:
    o = o1
elif prob_num == 2:
    o = o2
elif prob_num == 3:
    o = o3
    M = M3
elif prob_num == 4:
    o = o4
elif prob_num == 5:
    o = o5
    A = A5
elif prob_num == 6:
    o = o6
elif prob_num == 7:
    o = o7
    M= M7
elif prob_num == 8:
    o = o8
    M = M8
elif prob_num == 9:
    o = o9
elif prob_num == 10:
    o = o10
    M = M10

        
#### setting
generation_number = 100
population_number = 100
parameter_number = d
nfe = generation_number*population_number
test_num = 20
####################


g_fitness_list = []
fitness_list = np.zeros([nfe,1])
pbest_list = np.zeros([nfe,1])
gbest_list = np.zeros([nfe,1])

for t in range(test_num):

    x = np.zeros((population_number, parameter_number), dtype='float64')

    p_best = np.zeros([population_number, parameter_number], dtype='float64')
    g_best = np.zeros([parameter_number], dtype='float64')

    velocity = np.zeros((population_number, parameter_number), dtype='float64')
    x_fitness = np.zeros(population_number, dtype='float64')
    p_fitness = np.zeros(population_number, dtype='float64')
    g_fitness = 10**20

    for i in range(population_number):
        p_fitness[i] = g_fitness

    x = lhs(d, samples=population_number)
    for i in range(population_number):
         x[i,:] = x[i,:]*(x_max - x_min) + x_min
    
    for g in range(generation_number):
        for i in range(population_number):
            
    
            x_fitness[i] = fitness(x[i], prob_num, o = o, M = M, A = A)

            if x_fitness[i] < g_fitness:
                g_fitness = x_fitness[i]
                g_best[:] = x[i]

            if x_fitness[i] < p_fitness[i]:
                p_fitness[i] = x_fitness[i]
                p_best[i] = x[i]

            fitness_list[g*population_number + i] = x_fitness[i]
            pbest_list[g*population_number + i] = p_fitness[i]
            gbest_list[g*population_number + i] = g_fitness
            
            

            r1 = np.random.rand(d)
            r2 = np.random.rand(d)
            velocity[i] = w*(velocity[i] + C1 * r1 * (p_best[i]- x[i]) + C2 * r2 * (g_best - x[i]))
            
            for j in range(parameter_number):
                if velocity[i][j] > d_max[j]:
                    velocity[i][j] = d_max[j]
                    #print(velocity[i][j])
                    
                elif velocity[i][j] < -1*d_max[j]:
                    velocity[i][j] = -1*d_max[j]
                          
            for j in range(parameter_number):
                x[i][j] = x[i][j] + velocity[i][j]
    print(g_fitness)
    g_fitness_list.append(g_fitness)
    
g_fitness_array = np.array(g_fitness_list)
mean_result = np.mean(g_fitness_array)
best_result = np.min(g_fitness_array)
worst_result = np.max(g_fitness_array)
std = np.std(g_fitness_array)

print("best : %.6f, mean : %.6f, worst : %.6f, std : %.6f" %(best_result, mean_result, worst_result, std))

fitness_avg = np.average(fitness_list, axis=0)

k = np.arange(1,nfe+1)