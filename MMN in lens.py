#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Time    : 2018-11-20
# @Update  : 2019-04-24
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
import time
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

    # in this program, a represents the parameter M1 in the research letter.
    # in this program, b represents the parameter M2 in the research letter.
    # By manually adjusting a and b, we intercepted matrix B from matrix A.
    a = 140
    b = 210
    PE_cal_timer = 0
    B = A[a:b, ]
    iter = 0 # Recording the iteration
    PEgbest = 100
    time_start = time.time()                # Recording the system time when the program starts.
    PE_fitness = []                            # Recording the global PE value for each iteration
    timer = []                              # Recording the calculate time
    # MMN program, the step size and the limited domain corresponding to the paper.
    start_cal_time = time.time()            # Recording the calculate time
    for i in range(1, 6000, 5): # -0.003 to 0.003 mm with 0.000005 mm step
        p = (i-3000) * 0.000001
        for k in range(0, 20000, 1): #0 to 200 mm with 0.01 mm step
            x0 = k * 0.01 * -1
            for j in range(-100, 100, 1):  # -0.001 mm to 0.001 mm range with 0.00001 mm step
                y0 = j * 0.00001 * -1
                PE = PE_generator(p,x0,y0) # Calculating the PE of Gbest in each iteration.

                if PE < PEgbest:   # Determining if it is the current global best PE value than before
                    if (PEgbest-PE)/PEgbest <1e-12:   #setting the termination parameter to 1e-10%
                        print('The difference of two steps is',(PEgbest-PE))
                        break
                        # (PEgbest-PE)/PEgbest is sign as termination parameter.
                        #we believe that the termination parameter (a percentage number) is scientific and effective in
                        # the MMN software.
                        #This is because when the MMN algorithm is used to find the location of the target solution in
                        # the limited domain space, if the loop gradient and the termination parameter are set properly,
                        # the termination parameter can be triggered to help the users save time cost.
                        # If the termination parameter is too small, it may not be triggered and all limited domain may
                        # be traversed.
                        # If the termination parameter is too large, the convergence process may be terminated
                        # prematurely in a converged platform region.
                        # Therefore, a reasonable termination parameter is a key step to save time cost.
                        # In combination with the engineering practice (please refer to research paper),
                        # it is reasonable to set the percentage number to 1e-10%, which means 1e-12 in number.

                    PEgbest = PE   # Updating the global best PE value
                    PP = p         # Updating the global best PP
                    xx0 = x0       # Updating the global best xx0
                    yy0 = y0       # Updating the global best yy0
                    # Recording the global best PE value.
                    PE_fitness.append(PEgbest)
                    time_cal_temp = time.time() #Recording the time point when the best Fitness-value was changed.
                    timer.append(time_cal_temp-start_cal_time)
                #iter = iter + 1
        print(PEgbest)
    print('The final P is', PP)
    print('The final y0 is', yy0)
    print('The final x0 is', xx0)
    print('The PE of Gbest is', PEgbest)
    C = c_generator(PP,xx0,yy0)

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
    plt.legend(handles=[label_A, label_c], labels=['Unequal diameter glass tube profile curve', 'Intercepted part profile curve'], loc='upper left')
    plt.savefig('Intercepted part on the glass tube.png', dpi=600)
    plt.show()

    # Using matplotlib program package to draw the result figures.
    # Drawing the curve diagram of Global PE value and calculate_time.
    plt.figure(2)
    plt.title("")
    plt.xlabel("Run time / s", size=14)
    plt.ylabel("PE / mm", size=14)
    timer = np.array(timer)
    PE_fitness = np.array(PE_fitness)
    label_fitness, = plt.plot(timer, PE_fitness, color='b', linewidth=3)
    plt.legend(handles=[label_fitness], labels=['Global best PE value'], loc='upper right')
    plt.savefig('PE value with time.png', dpi=600)
    plt.show()

    #Saving the results to Files
    np.savetxt('Matrix_B_a'+str(a)+'_b'+str(b)+'.csv',B,delimiter=',') # Saving Matrix B to file
    np.savetxt('Matrix_C_a'+str(a)+'_b'+str(b)+'.csv',C,delimiter=',') # Saving Matrix C to file
