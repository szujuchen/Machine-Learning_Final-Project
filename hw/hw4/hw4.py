import numpy as np
from numpy import loadtxt
import random
from random import randint
import statistics
from statistics import mode
from statistics import mean
import math
from liblinear.liblinearutil import*
import itertools
from itertools import combinations
from itertools import combinations_with_replacement

#function for transformation of x
#function for calculate 0/1 error
def fourth_order(x):
    n = len(x)
    num = list(itertools.chain(range(0, n)))

    res = list(combinations_with_replacement(num, 2))
    for i in res:
      r = 1
      for j in i:
        r *= x[j]
      x = np.append(x, r, axis = None)
    '''
    res = list(combinations_with_replacement(num, 3))
    for i in res:
      r = 1
      for j in i:
        r *= x[j]
      x = np.append(x, r, axis = None)

    res = list(combinations_with_replacement(num, 4))
    for i in res:
      r = 1
      for j in i:
        r *= x[j]
      x = np.append(x, r, axis = None)
    '''
    x = np.insert(x, 0, 1, axis = None)
    return x

def calE(w, x, ans):
    Eout = float(0)
    for i in range(len(x)):
        sum = 0
        for j in range(len(x[i])):
            sum += w[j]*x[i][j]
        if(sum*ans[i] <= 0):
            Eout += 1
    Eout /= float(len(x))
    return Eout

#data prepare
file = open('train.txt', 'r')
data = []
for line in file:
    number_strings = line.split() 
    numbers = [float(n) for n in number_strings] 
    data.append(numbers)

x = np.array(data)
y = x[:, -1]
x = x[:, :-1]
fourth = fourth_order(x[0])
xtrain = np.empty((0, len(fourth)))
for i in x:
    xtrain = np.vstack([xtrain, fourth_order(i)])


file2 = open('test.txt', 'r')
data2 = []
for line in file2:
    number_strings = line.split() 
    numbers = [float(n) for n in number_strings] 
    data2.append(numbers)

x_test = np.array(data2)
y_test = x_test[:, -1]
x_test = x_test[:, :-1]
fourth = fourth_order(x_test[0])
xtest = np.empty((0, len(fourth)))
for i in x_test:
    xtest = np.vstack([xtest, fourth_order(i)])

lamb = np.array([-6, -3, 0, 3, 6])
min_Eout = 2
min_Ein = 2
bestout = 0
bestin = 0

#Q12 Q13 
min_Eout = 2
min_Ein = 2
for la in lamb:
    print("lambda = ", la, "::")
    ld = math.pow(10, la)
    c = 1/(2*ld)
    w = train(problem(y, xtrain), parameter('-q -s 0 -c ' + str(c) + ' -e 0.000001')).get_decfun()[0]
    eout = calE(w, xtest, y_test)
    print("eout = ", eout)
    if(eout <= min_Eout):
        min_Eout = eout
        bestout = la
    
    ein = calE(w, xtrain, y)
    print("ein = ", ein)
    if(ein <= min_Ein):
      min_Ein = ein
      bestin = la

print("min Eout with lambda = ",bestout)
print("min Ein with lambda = ",bestin)


#Q14 Q15 Q16
valla = []
Eout = []
Eout_ = []
for i in range(256):
  min_Eval = 2
  bestw = []
  bestla = 0
  randomlist = random.sample(range(0, len(xtrain)), 120)
  x_tr = np.empty((0, len(xtrain[0])))
  y_tr = np.empty((0, 1))
  x_val = np.empty((0, len(xtrain[0])))
  y_val = np.empty((0, 1))
  for i in range(0, len(x)):
    if i in randomlist:
      x_tr = np.vstack([x_tr, xtrain[i]])
      y_tr = np.vstack([y_tr, y[i]])
    else:
      x_val = np.vstack([x_val, xtrain[i]])
      y_val = np.vstack([y_val, y[i]])
  y_tr = y_tr.reshape(len(y_tr),)
  y_val = y_val.reshape(len(y_val),)
  for la in lamb:
    ld = math.pow(10, la)
    c = 1/(2*ld)
    w = train(problem(y_tr, x_tr), parameter('-q -s 0 -c ' + str(c) + ' -e 0.000001')).get_decfun()[0]
    eval = calE(w, x_val, y_val)
    
    if(eval <= min_Eval):
        min_Eval = eval
        bestla = la
        bestw = w
  
  valla.append(bestla)
  Eout.append(calE(bestw, xtest, y_test))

  ld = math.pow(10, bestla)
  c = 1/(2*ld)
  w_ = train(problem(y, xtrain), parameter('-q -s 0 -c ' + str(c) + ' -e 0.000001')).get_decfun()[0]
  Eout_.append(calE(w_, xtest, y_test))

print(mode(valla))
print(mean(Eout))
print(mean(Eout_))

#Q17
valla = []
Ecv = []
for i in range(256):
  min_Ecv = 2
  bestw = []
  bestla = 0
  randomlist = list(range(0, len(x)))
  random.shuffle(randomlist)
  
  for la in lamb:
    #print("lambda = ", la, "::")
    ld = math.pow(10, la)
    c = 1/(2*ld)
    ecv = float(0)
    for k in range(5):
      x_tr = np.empty((0, len(xtrain[0])))
      y_tr = np.empty((0, 1))
      x_val = np.empty((0, len(xtrain[0])))
      y_val = np.empty((0, 1))
      for j in range(0, len(x)):
        if j < 40*(k+1) and j >= (40)*k:
          x_val = np.vstack([x_val, xtrain[j]])
          y_val = np.vstack([y_val, y[j]])
        else:
          x_tr = np.vstack([x_tr, xtrain[j]])
          y_tr = np.vstack([y_tr, y[j]])
      y_tr = y_tr.reshape(len(y_tr),)
      y_val = y_val.reshape(len(y_val),)

      w = train(problem(y_tr, x_tr), parameter('-q -s 0 -c ' + str(c) + ' -e 0.000001')).get_decfun()[0]
      ecv += float(calE(w, x_val, y_val))
    ecv /= float(len(x)/40)

    if(ecv <= min_Ecv):
        min_Ecv = ecv
        bestla = la
        bestw = w
  
  valla.append(bestla)
  Ecv.append(min_Ecv)

print(mode(valla))
print(mean(Ecv))


#Q18 Q19
bestw_l1 = []
min_Eout = 2
for la in lamb:
    print("lambda = ", la, "::")
    ld = math.pow(10, la)
    c = 1/(ld)
    w = train(problem(y, xtrain), parameter('-q -s 6 -c ' + str(c) + ' -e 0.000001')).get_decfun()[0]
    eout = calE(w, xtest, y_test)
    print("eout = ", eout)
    if(eout <= min_Eout):
        min_Eout = eout
        bestout = la
        bestw_l1 = w

print("min Eout with lambda = ",bestout)
cnt = 0
for i in bestw_l1:
  if abs(i) <= math.pow(10, -6):
    cnt += 1
print(cnt)


#Q20 
bestw_l2 = []
min_Eout = 2
for la in lamb:
    print("lambda = ", la, "::")
    ld = math.pow(10, la)
    c = 1/(2*ld)
    w = train(problem(y, xtrain), parameter('-q -s 0 -c ' + str(c) + ' -e 0.000001')).get_decfun()[0]
    eout = calE(w, xtest, y_test)
    print("eout = ", eout)
    if(eout <= min_Eout):
      min_Eout = eout
      bestout = la
      bestw_l2 = w
    
print("min Eout with lambda = ",bestout)
cnt = 0
for i in bestw_l2:
  if abs(i) <= math.pow(10, -6):
    cnt += 1
print(cnt)
