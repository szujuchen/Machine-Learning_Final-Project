from libsvm.svmutil import *
import numpy as np
from numpy import loadtxt

file = open('train.txt', 'r')
data = []
for line in file:
    number_strings = line.split()
    numbers = []
    numbers.append(int(number_strings[0]))
    for i in range(1, 17):
      sub = number_strings[i].split(':')
      numbers.append(float(sub[1]))
    data.append(numbers)

x = np.array(data)
y_tr = x[:, 0]
x_tr = x[:, 1:]

print(x_tr.shape)
print(y_tr.shape)

file = open('test.txt', 'r')
data = []
for line in file:
    number_strings = line.split()
    numbers = []
    numbers.append(int(number_strings[0]))
    for i in range(1, 17):
      sub = number_strings[i].split(':')
      numbers.append(float(sub[1]))
    data.append(numbers)

x = np.array(data)
y_test = x[:, 0]
x_test = x[:, 1:]

print(x_test.shape)
print(y_test.shape)

def err(predict, ans):
  sum = 0
  num = len(predict)
  for i in range(0, num):
    if(int(predict[i]) != int(ans[i])):
      sum += 1
  return float(float(sum)/float(num))

#Q11
import math
pk = 1
tr = np.copy(y_tr)
tr[tr != pk] = -1

prob = svm_problem(tr, x_tr)
param = svm_parameter('-s 0 -t 0 -c 1')
model = svm_train(prob, param)
coef = model.get_sv_coef()
SV = model.get_SV()
W = np.zeros(16)
for i in range(0, len(coef)):
  for j in range(0, len(W)):
    #print(coef[i][0])
    #print(SV[i][j+1])
    W[j] += coef[i][0]*SV[i][j+1]

length = 0
for i in W:
  length += i*i
print(math.sqrt(length))

#Q12 #Q13
pk = [2, 3, 4, 5, 6]
Ein = []
nSV = []
for i in pk:
  tr = np.copy(y_tr)
  tr[tr != i] = -1
  tr[tr = i] = 1

  prob = svm_problem(tr, x_tr)
  param = svm_parameter('-s 0 -t 1 -d 2 -c 1 -r 1 -g 1')
  model = svm_train(prob, param)
  p_label, p_accu, p_val = svm_predict(tr, x_tr, model)
  num = model.get_nr_sv()
  Ein.append((err(p_label, tr), i))
  nSV.append((num, i))

min = __builtins__.min
print(Ein)
print(type(Ein))
print(min(Ein))
print(nSV)
print(min(nSV))

#Q14
C = [0.01, 0.1, 1, 10, 100]
Eout = []

tr = np.copy(y_tr)
tr[tr != 7] = -1
test = np.copy(y_test)
test[y_test != 7] = -1
print(test.shape)

for i in C:
  prob = svm_problem(tr, x_tr)
  param = svm_parameter('-s 0 -t 2 -g 1 -c '+ str(i))
  model = svm_train(prob, param)
  p_label, p_accu, p_val = svm_predict(test, x_test, model)
  Eout.append((err(p_label, test), i))

print(Eout)
print(min(Eout))

#Q15
gamma = [0.1, 1, 10, 100, 1000]
Eout = []

tr = np.copy(y_tr)
tr[tr != 7] = -1
test = np.copy(y_test)
test[y_test != 7] = -1

for i in gamma:
  prob = svm_problem(tr, x_tr)
  param = svm_parameter('-s 0 -t 2 -c 0.1 -g '+str(i))
  model = svm_train(prob, param)
  p_label, p_accu, p_val = svm_predict(test, x_test, model)
  Eout.append((err(p_label, test), i))

print(Eout)
print(min(Eout))

#Q16
from sklearn.model_selection import train_test_split
import statistics
from statistics import mode

gamma = [0.1, 1, 10, 100, 1000]
choose = []
print(len(gamma))

tr = np.copy(y_tr)
tr[tr != 7] = -1

for j in range(0, 500):
  x_trtr, x_tt, y_trtr, y_tt = train_test_split(x_tr, tr, test_size=200)
  mini = 2
  tar = 0
  print(j)
  for i in range(0, len(gamma)):
    #print(len(gamma))
    print(gamma[i])
    prob = svm_problem(y_trtr, x_trtr)
    param = svm_parameter('-s 0 -t 2 -c 0.1 -g '+str(gamma[i]))
    model = svm_train(prob, param)
    p_label, p_accu, p_val = svm_predict(y_tt, x_tt, model)
    Eval = err(p_label, y_tt)
    if(Eval < mini):
      mini = Eval
      tar = gamma[i]
  choose.append(tar)

print(len(choose))
print(choose)
print(mode(choose))

import collections

c = collections.Counter(choose)
print(c)

#data preprocess
import math
y_ada_tr = []
x_ada_tr = []
index_tr = []
y_ada_test = []
x_ada_test = []
index_test = []

cnt = 0
for i in range(0, len(y_tr)):
  if y_tr[i] == 11:
    y_ada_tr.append(1)
    x_ada_tr.append(x_tr[i])
    index_tr.append(cnt)
    cnt +=1
  elif y_tr[i] == 26:
    y_ada_tr.append(-1)
    x_ada_tr.append(x_tr[i])
    index_tr.append(cnt)
    cnt += 1

cnt = 0
for i in range(0, len(y_test)):
  if(y_test[i] == 11):
    y_ada_test.append(1)
    x_ada_test.append(x_test[i])
    index_test.append(cnt)
    cnt += 1
  elif y_test[i] == 26:
    y_ada_test.append(-1)
    x_ada_test.append(x_test[i])
    index_test.append(cnt)
    cnt += 1

x_ada_tr = np.array(x_ada_tr)
y_ada_tr = np.array(y_ada_tr)
x_ada_test = np.array(x_ada_test)
y_ada_test = np.array(y_ada_test)
index_tr = np.array(index_tr)
index_test = np.array(index_test)
print(x_ada_tr.shape)
print(y_ada_tr.shape)
print(x_ada_test.shape)
print(y_ada_test.shape)
x_sort = []
y_sort = []
index_sort = []

for i in range(0, len(x_ada_tr[0])):
  temp = x_ada_tr[:, i]
  x_sort.append(temp[temp.argsort()])
  y_sort.append(y_ada_tr[temp.argsort()])
  index_sort.append(index_tr[temp.argsort()])


x_sort = np.array(x_sort)
y_sort = np.array(y_sort)
index_sort = np.array(index_sort)
print(x_sort.shape)
print(y_sort.shape)

u = np.full((1,len(x_ada_tr)), float(1/len(x_ada_tr)), dtype = float)
s = np.array([1, -1])
thres = []
for i in range(0, len(x_sort)):
  sub = []
  sub.append(-math.inf)
  for j in range(0, len(x_sort[i])-1):
    if(x_sort[i][j] != x_sort[i][j+1]):
      sub.append(float((x_sort[i][j] + x_sort[i][j+1])/2))
  thres.append(sub)
thres = np.array(thres)
print(u.shape)
print(thres.shape)

n_feature = len(x_sort)
N = len(x_sort[0])
NN = len(x_ada_test)

def errada(besti, bests, bestth):
  ein = 0
  for k in range(0, N):
    ss = 1
    if(x_sort[besti][k] < bestth):
      ss = -1
    ss *= bests
    if(ss != y_sort[besti][k]):
      ein += 1
  ein = float(ein/N)
  return ein

#Q17
minein = 2
maxein = 0
Gin = np.zeros(N)
Gout = np.zeros(NN)
for t in range(0, 1000):
  print(t)
  mini = 2
  besti = 0
  bests = 0
  bestth = 0
  for i in range(0, n_feature):
    for sign in s:
      for th in thres[i]:
        einu = 0
        for j in range(0, N):
          ss = 1
          if(x_sort[i][j] < th):
            ss = -1
          ss *= sign
          if(ss != y_sort[i][j]):
            einu += u[0][index_sort[i][j]]
        einu = float(einu/N)
        if(einu < mini):
          mini = einu
          mineinu = einu
          besti = i
          bests = sign
          bestth = th
  ein = errada(besti, bests, bestth)
  if(ein < minein):
    minein = ein
  if(ein > maxein):
    maxein = ein

  sum = float(0)
  wr = float(0)
  for j in range(0, N):
    sum += u[0][index_sort[besti][j]]
    ss = 1
    if(x_sort[besti][j] < bestth):
      ss = -1
    ss *= bests
    if(ss != y_sort[besti][j]):
      wr += u[0][index_sort[besti][j]]
  epi = float(wr/sum)
  incor = math.sqrt(float((1-epi)/epi))
  cor = math.sqrt(float((epi)/(1-epi)))
  for j in range(0, N):
    ss = 1
    if(x_ada_tr[j][besti] < bestth):
      ss = -1
    ss *= bests
    if(ss != y_ada_tr[j]):
      u[0][j] *= incor
    else:
      u[0][j] *= cor
    Gin[j] += ss

  for j in range(0, NN):
    ss = 1
    if(x_ada_test[j][besti] < bestth):
      ss = -1
    ss *= bests
    Gout[j] += ss


print(minein)
print(maxein)

for j in range(0, N):
  if(Gin[j] >= 0):
    Gin[j] = 1
  else:
    Gin[j] = -1

for j in range(0, NN):
  if(Gout[j] >= 0):
    Gout[j] = 1
  else:
    Gout[j] = -1

print("einG:",err(Gin, y_ada_tr))
print("eoutG:",err(Gout, y_ada_test))