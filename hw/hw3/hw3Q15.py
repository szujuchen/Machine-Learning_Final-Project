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

x0 = 1
for row in data:
    row.insert(0, x0)

x = np.array(data)
y = x[:, -1]
x = x[:, :-1]


n = 0.001
Eavg = float(0)

for i in range(1000):
    wlin = np.zeros(len(x[0]))
    Ece = float(0)
    for j in range(800):
        m = randint(0, len(x)-1)
        er = np.matmul(wlin.transpose(), x[m])
        er = y[m]*er
        theta = float(1/(1 + pow(math.e, er)))
        final = n*theta
        final *= y[m]*x[m]
        wlin += final
    for j in range(len(x)):
        h = float(np.matmul(wlin.transpose(), x[j]))
        h *= -1
        h *= y[j]
        Ece += np.log(1+pow(math.e, h))
    Ece /= float(len(x))
    Eavg += Ece

Eavg /= float(1000)
print(Eavg)