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

mini_Ein = float(2)
max_Ein = float(0)
s_mini = 0
theta_mini = 0
i_mini = 0
s_max = 0
theta_max = 0
i_max = 0
for i in range(dimension):
    min_Ein = float(2)
    s_temp = 0
    theta_temp = 0
    i_temp = 0
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
                s_temp = s[j]
                theta_temp = theta[k]
                i_temp = i
    if(min_Ein > max_Ein):
        max_Ein = min_Ein
        s_max = s_temp
        theta_max = theta_temp
        i_max = i_temp
    if(min_Ein < mini_Ein):
        mini_Ein = min_Ein
        s_mini = s_temp
        theta_mini = theta_temp
        i_mini = i_temp

print(max_Ein)
print(mini_Ein)

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

mini_Eout = float(0)
for i in range(size):
        s = -1
        if(test_n[i][i_mini] - theta_mini > 0):
            s = 1
        if(s_mini*s*test_n[i][y] < 0):
            mini_Eout += 1

mini_Eout = float(mini_Eout)
mini_Eout /= float(size)
print(mini_Eout)

max_Eout = float(0)
for i in range(size):
        s = -1
        if(test_n[i][i_max] - theta_max > 0):
            s = 1
        if(s_max*s*test_n[i][y] < 0):
            max_Eout += 1

max_Eout = float(max_Eout)
max_Eout /= float(size)
print(max_Eout)

print(max_Eout - mini_Eout)




