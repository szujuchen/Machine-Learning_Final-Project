import numpy
from numpy import loadtxt
import random
from random import randint
import statistics
import math

def cal(w, x):
    sum = 0
    sum = float(sum)
    for i in range(11):
        sum += w[i]*x[i]
    return sum

#N initial
N = 256

#xn
file = open('hw1_train.txt', 'r')
data = []
for line in file:
    number_strings = line.split() 
    numbers = [float(n) for n in number_strings] 
    data.append(numbers)

#x0
x0 = 0.1126
for row in data:
    row.insert(0, x0)

data_n = numpy.array(data)
#scaling down
'''
for rows in data_n:
    rows[:11] *= 0.5
print(data_n[0][11])
'''
#print(len(data))
#print(len(data[0]))
#print(data)
#print(w)

sumEin = 0
sumEin = float(sumEin)
updates = []
w0s = []
w0x0s = []
for E in range(1000):
    #w0 initialization
    w = [0]*11
    #M iteration
    M = 4*N
    M = int(M)
    cnt = 0
    update = 0
    while True:
        x = randint(0, 255)
        #print(x)
        num = cal(w, data_n[x])
        #print(num)
        if (num >= 0 and data_n[x][11] < 0) or (num < 0 and data_n[x][11] > 0):
            for j in range(11):
                w[j] = w[j] + data_n[x][j]*data_n[x][11]
            cnt = 0
            update += 1
        else:
            cnt += 1
        if cnt >= M:
            break
    #15
    updates.append(update)
    #wf decided #16
    w0s.append(float(w[0]))
    w0x0s.append(float(w[0]*x0))
    #Ein calculate
    sum = 0
    for i in range(N):
        num = cal(w, data_n[i])
        if (num >= 0 and data_n[i][11] < 0) or (num < 0 and data_n[i][11] > 0):
            sum += 1
    sum = float(sum)
    sum = float(sum/N)
    #print(sum)
    sumEin += sum

print(sumEin/1000)
print(statistics.median(updates)) #15
print(statistics.median(w0s))
print(statistics.median(w0x0s))
