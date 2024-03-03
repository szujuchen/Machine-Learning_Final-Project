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

xtest = np.array(data2)
ytest = xtest[:, -1]
xtest = xtest[:, :-1]

n = 0.001
Eavg = float(0)

for i in range(1000):
    w0 = np.array([0.29070963, -0.04988084, 0.04893561, -0.08623605, -0.06658103, 0.10689752, -0.12356574,  0.09486241, 0.26696655, -0.15660245, -0.06382855])
    Ein = float(0)
    Eout = float(0)
    for j in range(800):
        m = randint(0, len(x)-1)
        er = np.matmul(w0.transpose(), x[m])
        er = y[m]*er
        theta = float(1/(1 + pow(math.e, er)))
        final = n*theta
        final *= y[m]*x[m]
        w0 += final
    for j in range(len(x)):
        if(y[j]*(np.matmul(w0.transpose(), x[j])) <= 0):
            Ein += 1
    Ein /= float(len(x))
    for j in range(len(xtest)):
        if(ytest[j]*(np.matmul(w0.transpose(), xtest[j])) <= 0):
            Eout += 1
    Eout /= float(len(xtest))
    Eavg += abs(Ein - Eout)

Eavg /= float(1000)
print(Eavg)