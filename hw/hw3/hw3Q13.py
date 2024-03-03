import numpy as np
from numpy import loadtxt
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
xt = x.transpose()

inv = np.matmul(xt, x)
inv = np.linalg.inv(inv)

wlin = np.matmul(inv, xt)
wlin = np.matmul(wlin, y)

print(wlin)

Esqr = float(0)
for i in range(len(x)):
    h = float(np.matmul(wlin.transpose(), x[i]))
    err = float(h - y[i])
    Esqr += float(pow(err, 2))
Esqr /= float(len(x))

print(Esqr)
