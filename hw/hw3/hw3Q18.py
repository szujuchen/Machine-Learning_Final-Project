import numpy as np
from numpy import loadtxt
import random
from random import randint
import statistics
import math

file = open('hw3train.txt', 'r')
data = []
for line in file:
    number_strings = line.split() 
    numbers = [float(n) for n in number_strings] 
    data.append(numbers)

file2 = open('hw3test.txt', 'r')
data2 = []
for line in file2:
    number_strings = line.split() 
    numbers = [float(n) for n in number_strings] 
    data2.append(numbers)

x0 = 1
for row in data:
    row.insert(0, x0)

for row in data2:
    row.insert(0, x0)

x = np.array(data)
y = x[:, -1]
x = x[:, :-1]
xt = x.transpose()

inv = np.matmul(xt, x)
inv = np.linalg.inv(inv)

wlin = np.matmul(inv, xt)
wlin = np.matmul(wlin, y)

xtest = np.array(data2)
ytest = xtest[:, -1]
xtest = xtest[:, :-1]

n = 0.001
Eavg = float(0)

Ein = float(0)
Eout = float(0)

for j in range(len(x)):
    if(y[j]*(np.matmul(wlin.transpose(), x[j])) <= 0):
        Ein += 1
Ein /= float(len(x))

for j in range(len(xtest)):
    if(ytest[j]*(np.matmul(wlin.transpose(), xtest[j])) <= 0):
        Eout += 1
Eout /= float(len(xtest))

Eavg = abs(Ein - Eout)
print(Eavg)