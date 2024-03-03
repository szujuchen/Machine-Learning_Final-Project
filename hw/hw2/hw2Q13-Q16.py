####Q13
import numpy as np
from numpy import loadtxt
import random
from random import randint
import statistics
from statistics import mean
import math

#create data set #keep the order x1 <= x2
time = 10000
dsize = 2
tau = 0

data = []
res = []
test = []
y = []

for i in range(time):
    data.append([])
    for j in range(dsize):
        data[i].append(random.uniform(-0.5, 0.5,))
    data[i].sort()
    res.append([])
    for j in range(dsize):
        res[i].append(-1)
        if(data[i][j] > 0):
            res[i][j] = 1
        flip = random.choices([1, -1], weights = (1-tau, tau), k = 1)
        res[i][j] *= flip[0]
    test.append(random.uniform(-0.5, 0.5))
    y.append(-1)
    if(test[i] > 0):
        y[i] = 1
    flip = random.choices([1, -1], weights = (1-tau, tau), k = 1)
    y[i] *= flip[0]
    

Eout_Ein = []
cal = []
#run the experiment 10000 times
for i in range(time):
    min_Ein = float(2)
    s_result = 0
    theta_result = 0

    theta = []
    theta.append(-math.inf)
    for k in range(1,dsize):
      if(data[i][k] != data[i][k-1]):
        theta.append(float((float(data[i][k-1]+data[i][k]))/2))

    s = []
    s.append(-1)
    s.append(1)

    #get the g
    for j in range(2):
        for k in range(len(theta)):
            sum_Ein = float(0)
            for l in range(dsize):
                sign = -1
                if(data[i][l] - theta[k] > 0):
                    sign = 1
                
                if(s[j]*sign*res[i][l] < 0):
                    sum_Ein += 1

            sum_Ein = float(float(sum_Ein)/float(dsize))
            if(sum_Ein < min_Ein):
                min_Ein = sum_Ein
                s_result = s[j]
                theta_result = theta[k]
            elif(sum_Ein == min_Ein):
                if((s[j]*theta[k]) < (s_result*theta_result)):
                    min_Ein = sum_Ein
                    s_result = s[j]
                    theta_result = theta[k]

    #do the Eout
    
    sum_Eout = 0
    for j in range(time):
        s = -1
        if(test[i]-theta_result > 0):
            s = 1
        if(s_result*s*y[i] < 0):
            sum_Eout += 1
    sum_Eout = float(sum_Eout)
    sum_Eout /= float(time)
    
    cal_Eout = min(abs(theta_result), 0.5)*(1-2*tau) +tau 
    cal.append(cal_Eout - min_Ein)
    Eout_Ein.append(sum_Eout - min_Ein)
print(mean(Eout_Ein))
print(mean(cal))