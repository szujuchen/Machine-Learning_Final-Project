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

for i in range(dimension):
    min_Ein = float(2)
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
    if(min_Ein > max_Ein):
        max_Ein = min_Ein

    if(min_Ein < mini_Ein):
        mini_Ein = min_Ein

print(max_Ein)
print(mini_Ein)
print(max_Ein - mini_Ein)