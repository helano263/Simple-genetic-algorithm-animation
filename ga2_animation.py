#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 17:03:12 2017

@author: Helano Oliveira

Description:  implement a SGA (Simple Genetic Algorithm). This code looks 
for the global maximum of a function. This code is the same code in the script
'trainning_ga1.py', with the difference that it generates an animation showing 
the population over the generations.

Last Update: 05 july 2017
"""

import matplotlib as mpl
# mpl.use('TkAgg')
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

# ========================== functions ========================================


def create_ind(length):
    """ 
    Creates an individual with binary cromossome of length 'length'
    INPUT: the lenght of the cromossome
    OUTPUT: the individual
    """
    gene = np.zeros((1,length), dtype=np.int)
    for i in range(length):
        gene[0,i] = np.random.randint(0,2, dtype=np.int)
    return gene

def create_pop(ind, gen):
    """ 
    Creates a population 
    INPUT: the number 'ind' of individuals and the number of genes of each ind    
    OUTPUT: the population
    """
    pop = np.zeros((ind, gen), dtype=np.int)
    for i in range(ind):
        pop[i,:] = pop[i,:] + create_ind(gen)
    return pop
    
def pop_b2d(p1):
    """ 
    Transform a populatio in binary to its correspondent in decimal
    INPUT: population in numpy array, where each row is an individual 
        represented as a binary cromossomme and every column of each individual is 
        one allele.
    OUTPUT: a numpy array in a column shape where the values are the decimal 
        correspondent for the binary cromossome
    """
    aux = p1*2**np.arange(p1.shape[1]-1, -1, -1)
    return np.vstack(aux.sum(axis=1))
    
def interp_pop(p2):
    """ 
    Interpolates the decimal values of the population between the lower and 
    upper limit of interest
    INPUT: the population represented in their decimal values
    OUTPUT: the population represented within the limits of interest
    """
    return lim_l + (lim_u - lim_l)*p2/(2**g - 1)

def evaluate_fitness(p3):
    """
    This function will evaluate the fitness of each individual given their 
    decimal correspondence
    INPUT: a numpy array in a column shape which every element is a decimal 
        representation of the individual
    OUTPUT: a numpy array the same shape as the input. Delivers their fitness
        based of the function model
    """
    return p3*np.sin(10*np.pi*p3) + 1

def rank_population(p4):
    """
    This function will rank the population based on their fitness. Highest 
    first. As an input, there must be a numpy matrix with the following 
    caracteristic concatenated side by side, respectively:
        
                   ith column
                   |          |       |        |         |    
                  v          v       v        v         v 
ith row ->  [    0:gene,    16  ,   17  ,    18   ,    19  ]
            [cromossome, decimal, interp. decimals, fitness]
            
    INPUT: matrix type where each row has the characteristic of one individual
    OUTPUT: Sorted population
    """
    return p4[np.flip(p4[:,19].argsort(), axis=0)]
    
def normalize_fitness(p5):
    """
    Individuals with negative fitness evaluation are bad. This function uses a
    linear interpolation to make all fitness in a positive interval defined by
    the the variables 'lower' and 'upper'
    INPUT: fitness of the population
    OUTPUT: fitness of the population in a positive interval
    """
    upper = 2
    lower = 0
    return lower + (upper - lower)*(p5 - p5.min())/(p5.max() - p5.min())

def acc_fitness(p5):
    """
    Calculates the accumulated normalized fitness and returns as a column array
    INPUT: normalized array of the individuals as a column array
    OUTPUT: accumulated normalized array of the individuals as a column array
    """
    return p5.sum(), np.array([p5.cumsum()]).T

def parent_selection(p5, num):
    """
    Decides which individuals will be selected to generate the offspring
    INPUT: accumulated fitness, number of dads
    OUTPUT: indices of the selected individuals among the population
    """
    selected = np.zeros((num, 1), dtype=np.int)
    for i in range(num):
        choice = fit_acc * np.random.random_sample()
        idx, y = np.where(p5 > choice)
        selected[i,0] = idx.min()
    return selected

def crossover(a, b):
    """
    Crossover the cromossomes of the parents to generate offspring. The 
    crossover will happen if the probability Pc is met. 
    INPUT: Two numpy arrays which carries the parents cromossomes
    OUTPUT: Two numpy arrays which carries the offspring cromossomes
    """
    
    if Pc >= np.random.random_sample():
        n = np.random.randint(1,g + 1)
        a1, a2 = np.concatenate((a[0, 0:n], b[0, n:]), axis=0), \
                 np.concatenate((b[0, 0:n], a[0, n:]), axis=0)
        return a1.reshape((1,g)), a2.reshape((1,g))
    else:
        return a, b

def mutation(a):
    """
    Operates a mutation of a bit in a cromossome of the given individual. A 
    mutation in a allele (bit) in a cromossome will happen if the propability
    Pm is met. Hence, if idx is True, the correspondent bit (in the cromossome)
    will change. 'idx' is a boolean array of the size of the cromossome. 
    INPUT: individual cromossome
    OUTPUT: individual cromossome mutated
    """
    idx = Pm >= np.random.random_sample(size=a.shape[1])
    return np.bitwise_xor(a,idx)

def new_generation(p5):
    """
    Produces a new generation doing crossover and mutation given the population
    INPUT: indexes of the selected parents
    OUTPUT: whole new population of cromossomes 
    """
    new_pop = np.empty((0, g), dtype=np.int)
    for i, j in zip(p5[::2], p5[1::2]):
        c, d = populacao_b[i, :], populacao_b[j, :]
        c, d = crossover(c, d)        
        c, d = mutation(c), mutation(d)        
        new_pop = np.concatenate((new_pop, c, d), axis=0)
    return new_pop

# ==================== animation functions ====================================
def init():

    line.set_data([], [])
    generation_text.set_text('')
    return line, generation_text
# animation function.  This is called sequentially
def animate(i):
    generation_text.set_text('Generation #%i' % (i+1))
    line.set_data(an_par2[i*N:i*N + N,0], an_par2[i*N:i*N + N,1])
    return line, generation_text

# ==================== plot a function of interest ============================
    

    
fig, ax1 = plt.subplots()
x = np.linspace(-1, 2, 1000)
y = x*np.sin(10*np.pi*x) + 1
ax1.plot(x, y)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_xlim([-1, 2])
ax1.axhline(1, color='red', lw=0.5)
ax1.set_facecolor('#e1e1e1')
fig.set_facecolor('#f1f1f1')
line, = ax1.plot([], [], ' ', marker='d')
generation_text = ax1.text(0.02, 0.90, '', transform=ax1.transAxes)

# ======================= main propram ========================================
# ======================= initial parameters ==================================
lim_l = -1      # lower limit of the function to be evaluated
lim_u = 2       # upper limit of the function to be evaluated
N = 30          # number of individuals in a population
g = 22          # numero de genes de cada individuo
Pc = 0.6        # crossver probability
Pm = 0.01       # Mutation probability
num_ger = 50    # number of generations

populacao_b = create_pop(N, g)
an_par2 = np.empty((0,2), dtype=np.float)
# ====================== iterate the generations ==============================
for i in range(num_ger):
    # print(i)
    populacao_d = pop_b2d(populacao_b)
    populacao_i = interp_pop(populacao_d)
    populacao_f = evaluate_fitness(populacao_i)
    table = np.concatenate((populacao_b, 
                        populacao_d,
                        populacao_i,
                        populacao_f), axis=1)
    table1 = rank_population(table)
    populacao_nf = normalize_fitness(np.array([table[:,-1]]).T) 
    fit_acc, populacao_c = acc_fitness(populacao_nf)
    select = parent_selection(populacao_c, 30)
    populacao_b = new_generation(select)            # new population
    # Store parameters for animation
    an_par1 = np.concatenate((populacao_i, populacao_f), axis=1)
    an_par2 = np.concatenate((an_par2,an_par1), axis=0)
    
# ==================== process the animation ==================================
# Plot the last population    
ax1.plot(populacao_i, populacao_f, ' ', color='green', marker='d', label = 'Last population')  
# Plot the animation
anim = animation.FuncAnimation(fig, animate, init_func=init, 
                               frames=num_ger, interval=500, blit=True)
ax1.legend(loc='lower left')
plt.show()
# To install ffmpeg, run the following command on terminal
# conda install -c menpo ffmpeg=3.1.3   

# To save the animation, uncomment the next line
anim.save('animation.mp4', fps=15)







