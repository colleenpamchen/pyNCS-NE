#!/bin/python
#-----------------------------------------------------------------------------
# Author: Emre Neftci
#
# Creation Date : Fri 05 May 2017 01:03:41 AM PDT
# Last Modified : 
#
# Copyright : (c) 
# Licence : GPLv2
#----------------------------------------------------------------------------- 
# from expPop import *
# M = np.eye(256, dtype = 'bool')
# kwargs = {'connection':M}
# conn1 = pyNCS.Connection(pop_exc1,pop_exc2,synapse='exc_fast',fashion='by_boolean_matrix', fashion_kwargs=kwargs)

from expPop import *

M1 = np.zeros((256,256)) # inh 
M2 = np.zeros((256,256)) # exc

for j in range(32,63):
        for i in range(0,15):
        	M1[i,j]=1
for j in range(64,95):
        for i in range(16,31):
        	M2[i,j]=1
for j in range(96,127):
        for i in range(64,79):
        	M1[i,j]=1
for j in range(128,159):
        for i in range(80,95):
        	M2[i,j]=1
for j in range(160,191):
        for i in range(128,143):
        	M1[i,j]=1
for j in range(192,223):
        for i in range(144,159):
        	M2[i,j]=1
for j in range(224,255):
        for i in range(192,207):
        	M1[i,j]=1

kwargs1 = {'connection': M1}
kwargs2 = {'connection': M2}
conn1 = pyNCS.Connection(pop_exc1,pop_exc2,synapse='exc_fast',fashion='by_boolean_matrix',fashion_kwargs=kwargs2)
conn2 = pyNCS.Connection(pop_exc1,pop_exc2,synapse='inh_fast',fashion='by_boolean_matrix',fashion_kwargs=kwargs1)



#                 M1[i,j]=1
#         for i in range(1,16,16):
#                 M1[i,j]=1
#         for i in range(2,16,16):
#                 M1[i,j]=1
#         for i in range(3,16,16):
#                 M1[i,j]=1
# for j in range(1,256,8):
# 	    for i in range(0,16,16):
#                 M1[i,j]=1
#         for i in range(1,16,16):
#         		M1[i,j]=1
#         for i in range(2,16,16):
#                 M1[i,j]=1
#         for i in range(3,16,16):
#                 M1[i,j]=1
# for j in range(2,256,8):
#         for i in range(0,16,16):
#                 M1[i,j]=1
#         for i in range(1,16,16):
#                 M1[i,j]=1
#         for i in range(2,16,16):
#                 M1[i,j]=1
#         for i in range(3,16,16):
#                 M1[i,j]=1
# for j in range(3,256,8):
#         for i in range(0,16,16):
#                 M1[i,j]=1
#         for i in range(1,16,16):
#                 M1[i,j]=1
#         for i in range(2,16,16):
#                 M1[i,j]=1
#         for i in range(3,16,16):
#                 M1[i,j]=1

# M3=zeros((256,256))
# for j in range(0,256,8):
#         for i in range(16):
#                 M3[i,j]=1
# for j in range(1,256,8):
#         for i in range(16):
#                 M3[i,j]=1
# for j in range(2,256,8):
#         for i in range(16):
#                 M3[i,j]=1
# for j in range(3,256,8):
#         for i in range(16):
#                 M3[i,j]=1
# for j in range(256):
# 		for i in range(4,256):
# 				M3[i,j]=1 


# M1=np.array(M1, dtype='bool')
# M3=np.array(M3, dtype='bool')
# M2 = ~M3
#M2 = ~M1

#M=np.eye(256,dtype='bool')
#kwargs={'connection':M}
#c = pyNCS.Connection(pop_exc1,pop_exc2,synapse='exc_fast',fashion='by_boolean_matrix',fashion_kwargs=kwargs)


