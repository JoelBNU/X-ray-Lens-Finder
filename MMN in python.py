# coding: utf-8
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Time    : 2018-11-29
# @Author  : Joel
# @File    :
# @Site    :
# @Software: Python Numpy
# @Version :
from numpy import *
from random import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from math import pi as PI
from math import sin, cos
from math import sqrt
import sys
from pylab import *
dir=os.getcwd()
print('The work dir is',dir)
dir_son=dir+'\data'
print('The data store dir is',dir_son)
global df
df=pd.read_csv(dir_son+'\\'+'data.csv')
dataflow=pd.DataFrame(columns=['x_position','data1','data2'])
global alt
alt=array(df[['Position','Data1','Data2']])
print(alt)
global PP
global xx0
global yy0
print(len(alt))
global a
global b
a=130
b=210
global alt_cut
alt_cut=alt[a:b,]
global FFgbest
FFgbest=1e10

def goalmatrix(P,x0,y0):
        alt_fit_value=np.zeros((len(alt_cut),3))
        alt_fit_value[:,0]=alt_cut[:,0]
        alt_fit_value[:,1]=-sqrt(2*P*(alt_fit_value[:,0]+x0))-y0
        alt_fit_value[:,2]=sqrt(2*P*(alt_fit_value[:,0]+x0))-y0 
        return alt_fit_value

def goalfunction(P,x0,y0):
        alt_fit_value=np.zeros((len(alt_cut),3))
        alt_fit_value[:,0]=alt_cut[:,0]
        alt_fit_value[:,1]=-sqrt(2*P*(alt_fit_value[:,0]+x0))-y0
        alt_fit_value[:,2]=sqrt(2*P*(alt_fit_value[:,0]+x0))-y0
        Value_Error_matrix=(alt_cut[:,1]-alt_fit_value[:,1])**2+(alt_cut[:,2]-alt_fit_value[:,2])**2
        Value_Error_sum=Value_Error_matrix.sum()
        return Value_Error_sum
    
import time 
time_start=time.time()
fitness=[]
for i in range(1,300,5):
    p=i*0.00001
    for j in range(1700,2700,5):
        y0=j*0.001*-1
        for k in range(0,4000,5):
            x0=k*0.01*-1
            fit_matrix=np.zeros((len(alt_cut),3))
            fit_matrix=goalmatrix(p,x0,y0)

            Value_Error_matrix=(alt_cut[:,1]-fit_matrix[:,1])**2+(alt_cut[:,2]-fit_matrix[:,2])**2
            Value_Error_sum=Value_Error_matrix.sum()

            FF=math.sqrt(Value_Error_sum/(b-a))
            if FF<FFgbest:
                FFgbest=FF
                PP=p
                xx0=x0
                yy0=y0

            fitness.append(FFgbest)
time_end=time.time()
print('total run cost is ',time_end-time_start)
print('P is',PP)
print('y0 is',yy0)
print('x0 is',xx0)

plt.figure(1)
plt.title("")
plt.xlabel("iteration", size=14)
plt.ylabel("FF of Gbest", size=14)
t = np.array([t for t in range(0, len(fitness))])
fitness = np.array(fitness)
plt.plot(t, fitness, color='b', linewidth=3)
plt.savefig('FF of Gbest.png', dpi=300)
plt.show()

fit_matrix=goalmatrix(PP,xx0,yy0)
plt.figure(2)  
plt.title("")  
plt.xlabel("X-axis / mm", size=14)  
plt.ylabel("Y-axis / mm", size=14)  

plt.plot(alt_cut[:,0],alt_cut[:,1], color='b',linewidth=3) 
plt.plot(alt_cut[:,0],alt_cut[:,2], color='b',linewidth=3) 
plt.plot(fit_matrix[:,0],fit_matrix[:,1], color='r',linewidth=3) 
plt.plot(fit_matrix[:,0],fit_matrix[:,2], color='r',linewidth=3) 
plt.savefig('detail.png', dpi=300)
plt.show()

plt.figure(3)  
plt.title("")  
plt.xlabel("X-axis / mm", size=14)  
plt.ylabel("Y-axis / mm", size=14)  

plt.plot(alt[:,0],alt[:,1], color='b',linewidth=3) 
plt.plot(alt[:,0],alt[:,2], color='b',linewidth=3) 
plt.plot(fit_matrix[:,0],fit_matrix[:,1], color='r',linewidth=3) 
plt.plot(fit_matrix[:,0],fit_matrix[:,2], color='r',linewidth=3) 
plt.savefig('whole.png', dpi=300)
plt.show()

