#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Time    : 2018-11-20
# @Update  : 2019-04-24
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
# to the first column data and Equation 1 (Eq. 1).
def c_generator(P, x0, y0):
    C = np.zeros((len(B), 3))
    C[:, 0] = B[:, 0] # copy the first column data in B
    C[:, 1] = -sqrt(2 * P * (C[:, 0] + x0)) - y0  # generate Y_downfit
    C[:, 2] = sqrt(2 * P * (C[:, 0] + x0)) - y0   # generate Y_upfit
    return C

# Defining function to generate the FF value, corresponding Eq. 2 in the research paper.
def FF_generator(P, x0, y0):
    C = np.zeros((len(B), 3))
    C[:, 0] = B[:, 0]
    C[:, 1] = -sqrt(2 * P * (C[:, 0] + x0)) - y0
    C[:, 2] = sqrt(2 * P * (C[:, 0] + x0)) - y0
    Value_Error_matrix = (B[:, 1] - C[:, 1]) ** 2 + (B[:, 2] - C[:, 2]) ** 2
    Value_Error_sum = Value_Error_matrix.sum()
    FF_value = math.sqrt(Value_Error_sum/float(len(B)))
    return FF_value



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
    # columns are the scanned lower profile Y_down and the upper profile Y_up.
    A = array(df[['Position', 'Data1', 'Data2']])
    print('The Matrix A is: \n',A)

    # Giving the parameters of Equation 1 (Eq. 1):（y+y0）**2=2P(x+x0)
    # x is an independent variable and y is a dependent variable.
    # in this program, PP represents the parameter P in Eq. 1
    # in this program, xx0 represents the parameter x0 in Eq. 1
    # in this program, yy0 represents the parameter y0 in Eq. 1

    # in this program, a represents the parameter M1 in the research paper.
    # in this program, b represents the parameter M2 in the research paper.
    # By manually adjusting a and b, we intercepted matrix B from matrix A.
    a = 130
    b = 210
    FFgbest = 1e15
    B = A[a:b, ]
    iter = 0 # Recording the iteration

    time_start = time.time()                # Recording the system time when the program starts.
    fitness = []                            # Recording the global FF value for each iteration
    # MMN program, the step and the range corresponding to the paper.
    for i in range(1, 300, 5):
        p = i * 0.00001
        for j in range(1700, 2700, 5):
            y0 = j * 0.001 * -1
            for k in range(0, 4000, 5):
                x0 = k * 0.01 * -1
                FF = FF_generator(p,x0,y0) # Calculating the FF of Gbest in each iteration.
                if FF < FFgbest:   # Determining if it is the current global best FF value than before
                    FFgbest = FF   # Updating the global best FF value
                    PP = p         # Updating the global best PP
                    xx0 = x0       # Updating the global best xx0
                    yy0 = y0       # Updating the global best yy0
                # Recording the global best FF value and print it at each iteration.
                fitness.append(FFgbest)
                print(FFgbest)
                iter = iter + 1

    print('The final P is', PP)
    print('The final y0 is', yy0)
    print('The final x0 is', xx0)
    print('The FF of Gbest is', FFgbest)
    C = c_generator(PP,xx0,yy0)

    # Recording the system time when the program finished and print the running time.
    time_end = time.time()
    print('total run cost is ',time_end-time_start)

    # Using matplotlib program package to draw the result figures.
    # Drawing the curve diagram of  Global FF value and iterations.
    plt.figure(1)
    plt.title("")
    plt.xlabel("iteration", size=14)
    plt.ylabel("FF of Gbest", size=14)
    t = np.array([t for t in range(0, iter)])
    fitness = np.array(fitness)
    label_fitness, = plt.plot(t, fitness, color='b', linewidth=3)
    plt.legend(handles=[label_fitness], labels=['FF of Gbest'], loc='upper right')
    plt.savefig('FF of Gbest.png', dpi=300)
    plt.show()

    # Using matplotlib program package to draw the result figures.
    # Drawing the details of the inner profile curves of the long lens in the intercepting area
    # (Matrix B) and the parabolic lens in the intercepting area (Matrix C).
    plt.figure(2)
    plt.title("")
    plt.xlabel("X-axis / mm", size=14)
    plt.ylabel("Y-axis / mm", size=14)
    label_B, = plt.plot(B[:,0],B[:,1], color='b',linewidth=3)
    label_B, = plt.plot(B[:,0],B[:,2], color='b',linewidth=3)
    label_C, = plt.plot(C[:,0],C[:,1], color='r',linewidth=3)
    label_C, = plt.plot(C[:,0],C[:,2], color='r',linewidth=3)
    plt.legend(handles = [label_B,label_C],labels= ['Real profile curve','Fitting profile curve'],loc = 'upper left')
    plt.savefig('detail.png', dpi=300)
    plt.show()

    # Using matplotlib program package to draw the result figures.
    # Drawing  the inner profile curves for the long lens (Matrix A) and the
    # parabolic lens found by PSO (Matrix C).
    plt.figure(3)
    plt.title("")
    plt.xlabel("X-axis / mm", size=14)
    plt.ylabel("Y-axis / mm", size=14)
    label_A, = plt.plot(A[:,0],A[:,1], color='b',linewidth=3)
    label_A, = plt.plot(A[:,0],A[:,2], color='b',linewidth=3)
    label_c, = plt.plot(C[:,0],C[:,1], color='r',linewidth=3)
    label_c, = plt.plot(C[:,0],C[:,2], color='r',linewidth=3)
    plt.legend(handles=[label_A, label_c], labels=['Real all profile curve', 'Fitting profile curve'], loc='upper left')
    plt.savefig('whole.png', dpi=300)
    plt.show()

    #Saving the results to Files
    np.savetxt('Matrix_B_a'+str(a)+'_b'+str(b)+'.csv',B,delimiter=',') # Saving Matrix B to file
    np.savetxt('Matrix_C_a'+str(a)+'_b'+str(b)+'.csv',C,delimiter=',') # Saving Matrix C to file
