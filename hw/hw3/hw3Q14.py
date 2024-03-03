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
    Esqr = float(0)
    for j in range(800):
        m = randint(0, len(x)-1)
        er = float(np.matmul(wlin.transpose(), x[m]))
        er -= y[m]
        er *= -2
        er *= n
        wlin += er*x[m]
    for j in range(len(x)):
        h = float(np.matmul(wlin.transpose(), x[j]))
        Esqr += float(pow((h-y[j]),2))
    Esqr /= float(len(x))
    Eavg += Esqr

Eavg /= float(1000)
print(Eavg)