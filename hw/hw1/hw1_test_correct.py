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

#N
N = 256

#xn
file = open('hw1_train.txt', 'r')

data = []

for line in file:

  number_strings = line.split() 

  numbers = [float(n) for n in number_strings] 

  data.append(numbers)

x0 = 0.1126
for row in data:
    row.insert(0, x0)

data_n = numpy.array(data)

w0x0s = []
for E in range(1000):
    #w0 initialization
    w = [0]*11
    #M iteration
    M = int(4*N)
    cnt = 0
    while True:
        #random pick a xn
        x = randint(0, 255)
        #calculate the wfxn
        num = cal(w, data_n[x])
      	#check sign
        if (num >= 0 and data_n[x][11] < 0) or (num < 0 and data_n[x][11] > 0):
            for j in range(11):
                w[j] = w[j] + data_n[x][j]*data_n[x][11]
            cnt = 0
        else:
            cnt += 1
        if cnt >= M: #stop at M correct consecutive check
            break
    #put this round's w0*x0 to the list
    w0x0s.append(float(w[0]*x0))

#get the medain number of upadtes
print(statistics.median(w0x0s))