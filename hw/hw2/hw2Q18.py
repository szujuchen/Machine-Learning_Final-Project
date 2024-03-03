import numpy as np
from numpy import loadtxt
import random
from random import randint
import statistics
from statistics import mean
import math

file = open('hw2_train.txt', 'r')
data = []
for line in file:
    number_strings = line.split() 
    numbers = [float(n) for n in number_strings] 
    data.append(numbers)

data_n = np.array(data)
np.sort(data_n, axis = 0)

dimension = len(data[0]) - 1
size = len(data)
y = len(data[0]) - 1

min_Ein = float(2)
s_result = 0
theta_result = 0
i_result = 0
for i in range(dimension):

    theta = np.zeros(size)
    theta[0] = -math.inf
    for k in range(1,size):
        theta[k] = float((float(data_n[k-1][i]+data_n[k][i]))/2)
    
    s = np.zeros(2)
    s[0] = -1
    s[1] = 1

    for j in range(2):
        for k in range(size):
            sum_Ein = float(0)
            for l in range(size):
                sign = -1
                if(data_n[l][i] - theta[k] > 0):
                    sign = 1
                
                if(s[j]*sign*data_n[l][y] < 0):
                    sum_Ein += 1
            sum_Ein = float(sum_Ein)
            sum_Ein /= size

            if(sum_Ein < min_Ein):
                min_Ein = sum_Ein
                s_result = s[j]
                theta_result = theta[k]
                i_result = i

#print(min_Ein)

file = open('hw2_test.txt', 'r')
test = []
for line in file:
    number_strings = line.split() 
    numbers = [float(n) for n in number_strings] 
    test.append(numbers)

test_n = np.array(test)

dimension = len(test_n[0]) - 1
size = len(test_n)
y = len(test_n[0]) - 1

Eout = float(0)
for i in range(size):
        s = -1
        if(test_n[i][i_result] - theta_result > 0):
            s = 1
        if(s_result*s*test_n[i][y] < 0):
            Eout += 1

Eout = float(Eout)
Eout /= float(size)
print(Eout)




