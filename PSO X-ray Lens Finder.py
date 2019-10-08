#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Time    : 2018-11-20
# @Update  : 2019-04-22
# @Update  : 2019-08-16
# @Update  : 2019-09-15 leasted one
# @Author  : Joel
# @File    :
# @Site    :
# @Software: Python / Pandas Numpy Matplotlib
# @Version : Python 3.6

# importing the necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from math import pi as PI
from math import sqrt
from pylab import *

# Please refer to the research paper for more information about Matrix A, B and C.
# Defining function to generate the Matrix C.
# In matrix C, the first column data are a copy of the first column data in matrix B.
# The second and third columns contain the Y_downfit and Y_upfit data generated according
# to the first column data and Equation 2 (Eq. 2).
def c_generator(P, x0, y0):
    C = np.zeros((len(B), 3))
    C[:, 0] = B[:, 0] # copy the first column data in B
    C[:, 1] = -sqrt(2 * P * (C[:, 0] + x0)) - y0  # generate Y_downfit
    C[:, 2] = sqrt(2 * P * (C[:, 0] + x0)) - y0   # generate Y_upfit
    return C

# Defining function to generate the PE value, corresponding Eq. 5 in the research paper.
def PE_generator(P, x0, y0):
    C = np.zeros((len(B), 3))
    C[:, 0] = B[:, 0]
    C[:, 1] = -sqrt(2 * P * (C[:, 0] + x0)) - y0
    C[:, 2] = sqrt(2 * P * (C[:, 0] + x0)) - y0
    Value_Error_matrix = (B[:, 1] - C[:, 1]) ** 2 + (B[:, 2] - C[:, 2]) ** 2
    Value_Error_sum = Value_Error_matrix.sum()
    PE_value = math.sqrt(Value_Error_sum/(2*float(len(B))))
    return PE_value

# Defining the PSO program
class PSO():
    # Program initialization,
    # input: Particle number, Dimension, Number of iterations, inertia weight, acceleration factors.
    def __init__(self, Pnum, dim, max_iter,w_start,w_end, c1, c2,gfit,gfit_temp,dimension_max, dimension_coefficient):
        self.Pnum = Pnum                                    # Particle number
        self.dim = dim                                      # Dimension
        self.max_iter = max_iter                            # Number of iterations
        self.w_start = w_start                              # initial inertia weight
        self.w_end = w_end                                  # ultimate inertia weight
        self.c1 = c1                                        # acceleration factor
        self.c2 = c2                                        # acceleration factor
        self.position = np.zeros((self.Pnum, self.dim))     # the initial space for position of all particles
        self.velocity = np.zeros((self.Pnum, self.dim))     # the initial space for velocity of all particles
        self.pbest = np.zeros((self.Pnum, self.dim))        # best position of each particle
        self.gbest = np.zeros((1, self.dim))                # global best position of all particles
        self.pfit = np.zeros(self.Pnum)                     # the initial space for FF value of all particles
        self.gfit = gfit                                    # the initial global FF value in this program
        self.gfit_temp = gfit_temp                          # the initial global FF temp value for end the optimization
                                                            # if the fitness-value does not change by a given %.
        self.dimension_max = dimension_max                  # the maximum value of each dimension, it can be manually
                                                            # adjusted according to the data to be analyzed.
        self.dimension_coefficient = dimension_coefficient  # the coefficient to set the maximum velocity according the
                                                            # maximum value of each dimension.
        self.PP = 0                                         # the initial P value
        self.xx0 = 0                                        # the initial x0 value
        self.yy0 = 0                                        # the initial y0 value

    # Defining function to obtain the value of PE.
    def PE_value_function(self, para1, para2, para3):
        PE_v = PE_generator(para1, para2, para3)
        return PE_v

    # Defining function to generate the C matrix.
    def C_matrix(self):
        c_matrix = c_generator(self.PP, self.xx0, self.yy0)
        return c_matrix

    # Defining function to initial the particles population
    def init_Population(self):
        import random
        for i in range(self.Pnum):
            for j in range(self.dim):
                # Assigning initial values ​​to each dimension of each particle
                self.position[i][j] = random.uniform(-self.dimension_max, self.dimension_max)
                # Assigning initial values ​​to each dimension of each particle
                self.velocity[i][j] = random.uniform(-self.dimension_max * self.dimension_coefficient,
                                                     self.dimension_max * self.dimension_coefficient)

            self.pbest[i] = self.position[i]              # Assigning initial values ​​to each dimension of each particle
            para1 = self.position[i, 0]                   # Obtaining P value of particle i
            para2 = self.position[i, 1]                   # Obtaining x0 value of particle i
            para3 = self.position[i, 2]                   # Obtaining y0 value of particle i
            PE = self.PE_value_function(para1, para2, para3)  # calculating the PE value of each particle
            self.pfit[i] = PE                                 # Recording the PE value of each particle
            if (PE < self.gfit):                       # Determining if it is the current best PE value of all particles
                self.gfit = PE                         # Updating the global best PE value
                self.gbest = self.position[i]          # Updating the global best position

    # Defining function to iterate the particles population
    def iterator(self):
        import random
        PE_fitness = []  # Recording the global PE value for each iteration
        timer = []    # Recording the calculate time
        time_cal_start = time.time() #Record the start time
        judgement = 0
        for t in range(self.max_iter):

            for i in range(self.Pnum):
                para1 = self.position[i, 0]        # Obtaining P value of particle i
                para2 = self.position[i, 1]        # Obtaining x0 value of particle i
                para3 = self.position[i, 2]        # Obtaining y0 value of particle i
                PE = self.PE_value_function(para1, para2, para3)      # calculating the PE value of each particle
                # Determining if it is the current best PE value of the particle
                if (PE < self.pfit[i]):
                    self.pfit[i] = PE              # Updating the best PE value of the particle
                    self.pbest[i] = self.position[i]  # Updating the best position of the particle
                    # Determining if it is the current global best PE value of all the particles
                    if (self.pfit[i] < self.gfit):
                        self.gbest = self.position[i] # Updating the global best position
                        self.gfit = self.pfit[i]      # Updating the global best PE value

            # Explaining the inertia weight in Eq. 3 in the research paper.
            # In PSO program, inertia weight (calculate_w) embodies the ability of particles to inherit previous
            # velocity. It can be selected between 0-1, larger weights are good for global search,
            # and smaller weights are good for local search.
            # Usually, we can fix the inertia weight (calculate_w) to a constant, such as 0.8.
            # In order to better balance the global search ability and the local search ability,
            # the inertia weight (calculate_w) can be gradually reduced in the calculation process,
            # and there are various methods to reduce the inertia weight (calculate_w),
            # such as linearly decreasing or nonlinearly decreasing.
            # In this program, we reduce the inertia weight (calculate_w) nonlinearly by the following formula.
            # The w_start is the initial calculate_w and set to 0.9;
            # the w_end is the end calculate_w and set to 0.4;
            # the t is the iteration number in the current running process;
            # the self.max_iter is the maximum iteration.
            # Discussion about inertia weights is no longer in the research paper.
            calculate_w = self.w_start - (self.w_start - self.w_end) * (t / self.max_iter) ** 2
            for i in range(self.Pnum):
                for j in range(self.dim):
                    # calculating the velocity of each particle and each dimension
                    self.velocity[i][j] = calculate_w * self.velocity[i][j] + self.c1 * random.uniform(0,1) * \
                                          (self.pbest[i][j] - self.position[i][j]) + self.c2 * random.uniform(0,1)\
                                          * (self.gbest[j] - self.position[i][j])
                    # If the velocity of any dimension exceeds the maximum velocity in restricted domain, limit it.
                    # the restricted domain was related to the initial velocity of particles population,
                    # refer to the research paper.
                    if self.velocity[i][j] > 2 * self.dimension_coefficient * self.dimension_max:
                        self.velocity[i][j] = 2 * self.dimension_coefficient * self.dimension_max
                    if self.velocity[i][j] < -2 * self.dimension_coefficient * self.dimension_max:
                        self.velocity[i][j]= -2 * self.dimension_coefficient * self.dimension_max
                # Updating the position of the particle i
                self.position[i] = self.position[i] + self.velocity[i]
            # Recording the global best PE value and print it at each iteration.
            time_cal_temp = time.time()

            PE_fitness.append(self.gfit)
            timer.append(time_cal_temp-time_cal_start)
            self.gfit_temp = self.gfit   #Update the Temp fitness-value

        # When the program finished, print the final results.
        self.PP = self.gbest[0]
        self.xx0 = self.gbest[1]
        self.yy0 = self.gbest[2]
        print('The Global Best Position is ', self.gbest, self.PP, self.xx0, self.yy0)
        print('The gfit is', self.gfit)
        return PE_fitness,timer



if __name__=='__main__':
    # Checking the working directory and printing
    dir = os.getcwd()
    print('The work dir is', dir)

    # Getting the data storage directory and printing
    dir_son = dir + '\data_20181009'  # Manually fill in the data file to be analyzed
    print('The data store dir is', dir_son)

    # Using pandas program package read the data Files
    df = pd.read_csv(dir_son + '\\' + 'data.csv')
    dataflow = pd.DataFrame(columns=['x_position', 'data1', 'data2'])

    # Matrixing the data and Generating the matrix A corresponding in the research paper.
    # The first column is the scan position data. The data in the second and third
    # columns are the scanned down profile Y_down and the up profile Y_up.
    A = array(df[['Position', 'Data1', 'Data2']])
    print('The Matrix A is: \n',A)

    # Giving the parameters of Equation 2 (Eq. 2):（y+y0）**2=2P(x+x0)
    # x is an independent variable and y is a dependent variable.
    # in this program, PP represents the parameter P in Eq. 2
    # in this program, xx0 represents the parameter x0 in Eq. 2
    # in this program, yy0 represents the parameter y0 in Eq. 2

    # in this program, a represents the parameter M1 in the research paper.
    # in this program, b represents the parameter M2 in the research paper.
    # By manually adjusting a and b, we intercepted matrix B from matrix A.
    a = 140
    b = 210
    B = A[a:b, ]

    time_start = time.time()                # Recording the system time when the program starts.
    iter = 1000                             # Setting the value of iteration.
    my_pso = PSO(50,3,iter,0.9,0.4,2,2,1e15,1e15,10000,0.2)     # Instantiating the 'PSO class' into my_pso.
    # The number of particles were set to 50 and the dimension was 3;
    # the initial gfit was set to 1e15, dimension maximum was set to 10000, the coefficient was set to 0.2;
    # the inertia weight and acceleration factors were set.

    # initial the population of my_pso
    my_pso.init_Population()

    # running the program
    PE_fitness,timer = my_pso.iterator()

    # Obtaining the Matrix C
    C = my_pso.C_matrix()

    # Recording the system time when the program finished and print the running time.
    time_end = time.time()
    print('total run cost is ',time_end-time_start)

    # Using matplotlib program package to draw the result figures.
    # Drawing  the inner profile curves for the unequal diameter glass tube (Matrix A) and the
    # parabolic lens found by PSO (Matrix C).
    plt.figure(1)
    plt.title("")
    plt.xlabel("X-axis / mm", size=14)
    plt.ylabel("Y-axis / mm", size=14)
    label_A, = plt.plot(A[:,0],A[:,1], color='k',linewidth=4)
    label_A, = plt.plot(A[:,0],A[:,2], color='k',linewidth=4)
    label_c, = plt.plot(C[:,0],C[:,1], 'r|',linewidth=4,markersize=10)
    label_c, = plt.plot(C[:,0],C[:,2], 'r|',linewidth=4,markersize=10)
    plt.legend(handles=[label_A, label_c], labels=['Varying diameter glass tube profile curve', 'Cut part profile curve'], loc='upper left')
    plt.savefig('Intercepted part on the glass tube.png', dpi=600)
    plt.show()

    # Using matplotlib program package to draw the result figures.
    # Drawing the curve diagram of  Global FF value and calculate_time.
    plt.figure(2)
    plt.title("")
    plt.xlabel("Run time / s", size=14)
    plt.ylabel("PE / mm", size=14)
    timer = np.array(timer)
    PE_fitness = np.array(PE_fitness)
    label_fitness, = plt.plot(timer, PE_fitness, color='b', linewidth=4)
    plt.legend(handles=[label_fitness], labels=['Global best PE value'], loc='upper right')
    plt.savefig('PE value with time.png', dpi=600)
    plt.show()

    #Saving the results to Files
    np.savetxt('Matrix_B_a'+str(a)+'_b'+str(b)+'.csv',B,delimiter=',') # Saving Matrix B to file
    np.savetxt('Matrix_C_a'+str(a)+'_b'+str(b)+'.csv',C,delimiter=',') # Saving Matrix C to file
