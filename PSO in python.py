# coding: utf-8
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Time    : 2018-11-20
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
dir_son=dir+'\data_20181206_1'
print('The data store dir is',dir_son)
global df
df=pd.read_csv(dir_son+'\\'+'data.csv')
#print(df)
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

class PSO():
    def __init__(self,Pnum,dim,max_iter):
        self.w = 0.8
        self.c1 = 2
        self.c2 = 2
        self.r1 = 0.8
        self.r2 = 0.3
        self.Pnum = Pnum
        self.dim = dim
        self.max_iter = max_iter
        self.position = np.zeros((self.Pnum,self.dim))
        self.velocity = np.zeros((self.Pnum,self.dim))
        self.pbest = np.zeros((self.Pnum,self.dim))
        self.gbest = np.zeros((1,self.dim))
        self.pfit = np.zeros(self.Pnum)
        self.gfit = 1e15
        self.PP=0
        self.xx0=0
        self.yy0=0
        
                               
    def function(self,para1,para2,para3):
        a=goalfunction(para1,para2,para3)
        return a
    
    def result_function(self):
        b = goalmatrix(self.PP,self.xx0,self.yy0)
        return b       
    
    def init_Population(self):
        import random
        for i in range(self.Pnum):
            for j in range(self.dim):
                self.position[i][j] = random.uniform(-10000,10000)
                self.velocity[i][j] = random.uniform(-10000,10000)
              
            self.pbest[i] = self.position[i]
            para1 = self.position[i,0]
            para2 = self.position[i,1]
            para3 = self.position[i,2]
            temp = self.function(para1,para2,para3)
            FF=math.sqrt(temp/(b-a)) 
            self.pfit[i]=FF
            if (FF < self.gfit):
                self.gfit = FF
                print(self.gfit)
                self.gbest = self.position[i]
                

                
    def iterator(self):
        fitness=[]
        for t in range(self.max_iter):
            for i in range(self.Pnum):
                para1 = self.position[i,0]
                para2 = self.position[i,1]
                para3 = self.position[i,2]
                temporary = self.function(para1,para2,para3)
                FF=math.sqrt(temporary/(b-a)) 
                if (FF < self.pfit[i]):
                    self.pfit[i] = FF
                    self.pbest[i] = self.position[i]
                    if (self.pfit[i] < self.gfit):
                        self.gbest = self.position[i]
                        self.gfit = self.pfit[i]

                        
            for i in range(self.Pnum):
                self.velocity[i] = self.w*self.velocity[i] + self.c1*self.r1*(self.pbest[i]-self.position[i]) +                                    self.c2*self.r2*(self.gbest-self.position[i])  #这里的gbest 是唯一值，产生bug的地方
                self.position[i] = self.position[i] + self.velocity[i]
            
            fitness.append(self.gfit)
            print(self.gfit)            
        
        self.PP=self.gbest[0]
        self.xx0=self.gbest[1]
        self.yy0=self.gbest[2]
        print('The Global Best Position is ',self.gbest,self.PP,self.xx0,self.yy0)
        return fitness
    


global iter
iter=1000
time_start=time.time()
my_pso = PSO(50,3,iter)
my_pso.init_Population()
fitness = my_pso.iterator()
time_end=time.time()
print('total run cost is ',time_end-time_start)

plt.figure(1)
plt.title("")
plt.xlabel("iteration", size=14)
plt.ylabel("FF of Gbest", size=14)
t = np.array([t for t in range(0, iter)])
fitness = np.array(fitness)
plt.plot(t, fitness, color='b', linewidth=3)
plt.savefig('FF of Gbest.png', dpi=300)
plt.show()

goal_matrix=my_pso.result_function()

plt.figure(2)  
plt.title("")  
plt.xlabel("X-axis / mm", size=14)  
plt.ylabel("Y-axis / mm", size=14)  

plt.plot(alt_cut[:,0],alt_cut[:,1], color='b',linewidth=3) 
plt.plot(alt_cut[:,0],alt_cut[:,2], color='b',linewidth=3) 
plt.plot(goal_matrix[:,0],goal_matrix[:,1], color='r',linewidth=3) 
plt.plot(goal_matrix[:,0],goal_matrix[:,2], color='r',linewidth=3) 
plt.savefig('detail.png', dpi=300)
plt.show()

plt.figure(3)  
plt.title("")  
plt.xlabel("X-axis / mm", size=14)  
plt.ylabel("Y-axis / mm", size=14)  

plt.plot(alt[:,0],alt[:,1], color='b',linewidth=3) 
plt.plot(alt[:,0],alt[:,2], color='b',linewidth=3) 
plt.plot(goal_matrix[:,0],goal_matrix[:,1], color='r',linewidth=3) 
plt.plot(goal_matrix[:,0],goal_matrix[:,2], color='r',linewidth=3) 
plt.savefig('whole.png', dpi=300)
plt.show()    

np.savetxt('alt_cut_a130_b210.csv',alt_cut,delimiter=',')   
np.savetxt('matrix_a130_b210.csv',goal_matrix,delimiter=',')  

