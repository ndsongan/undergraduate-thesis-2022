#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 14:24:00 2021

@author: thucnguyen
"""

from random import randint

ds = 10 #number of observed data
p = 4 #number of status

#generate a random list of data D
D = [0 for i in range(ds)]
D[0] = randint(50, 100)
for i in range(1,ds):
    D[i] = D[i-1] + randint(0,20)

#C is list of difference data
C = [0 for i in range(ds)]
for i in range(1, ds):
    t = (D[i]-D[i-1])
    if t <= 3:
        C[i] = 0 #status = decrease = 0
    else:
        if t <= 8:
            C[i] = 1 #status = slightly increase = 1
        else:
            if t <= 12:
                C[i] = 2 #status = normal increase = 2
            else:
                C[i] = 3 #status = sharply increase

#compute the transition matrix P
P = [[0 for j in range(p)] for i in range(p)]
P = [[0.166667, 0.428571, 0, 0], [0.83333, 0.428571, 0.5, 0.5], [0, 0.071429, 0, 0.5],
     [0, 0.071429, 0.5, 0]]

for i in range(1, ds):
    P[C[i]][C[i-1]] += 1
for i in range(p):
    print(P[i])
S = [0 for i in range(p)]
for j in range(p):
    for i in range(p):
        S[j] += P[i][j]
for i in range(p):
    for j in range(p):
        if S[j] != 0:
            P[i][j] = P[i][j]/S[j]

#print to test
print('Transition matrix')
for i in range(p):
    print(P[i])
print(S)

#initiation distribution a
a = [0, 0.5, 0.5, 0]
n = 2 #number of steps

#compute the matrix power Pn = P^n
Pn=P
s = len(P)
for t in range(n-1):
    R=[[0 for j in range(s)] for i in range(s)]
    for i in range(s):
        for j in range(s):
            for k in range(s):
                R[i][j] += Pn[i][k]*P[k][j]
Pn = R

#predict nth step an = a^n
an = [0 for i in range(len(a))]
for i in range(len(Pn)):
    for j in range(len(a)):
        an[i] = an[i] + Pn[i][j]*a[j]

print('distribution vector:', an)