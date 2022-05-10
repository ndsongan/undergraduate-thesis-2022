#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 20:57:50 2021

@author: thucnguyen
"""

import matplotlib.pyplot as plt 
P = [[0.166667, 0.428571, 0.0, 0.0], [0.83333, 0.428571, 0.5, 0.5], 
     [0.0, 0.071429, 0.0, 0.5],[0.0, 0.071429, 0.5, 0.0]]

a = [0.1, 0.6, 0.2, 0.1]
#a = [0, 1, 0, 0]

n = 10
Pn = P

s1 = [a[0]]
s2 = [a[1]]
s3 = [a[2]]
s4 = [a[3]]
print('Distribution Vector')
for t in range(n - 1):
    R = [[0 for j in range(len(P))] for i in range(len(P))]
    for i in range(len(P)):
        for j in range(len(P)):
            for k in range(len(P)):
                R[i][j] += P[i][k] * Pn[k][j]
    Pn = R
    an = [0 for i in range(len(a))]
    for i in range(len(Pn)):
        for j in range(len(a)):
            an[i] = an[i] + Pn[i][j] * a[j]
    s1.append(round(an[0], 3))
    s2.append(round(an[1], 3))
    s3.append(round(an[2], 3))
    s4.append(round(an[3], 3))
    print("Day ", t, ": ", an)
    
print('\n')
print('Distribution Matrix')
for i in range(len(Pn)):
    print(Pn[i])
    
t = [i for i in range(n)]
plt.plot(t, s1, label = r'$S_1$')
plt.plot(t, s2, label = r'$S_2$')
plt.plot(t, s3, label = r'$S_3$')
plt.plot(t, s4, label = r'$S_4$')
plt.ylabel('Distribution vector values')
plt.xlabel('Day')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
plt.title(r'$ \alpha $ = (' + str(a[0]) + ', ' + str(a[1]) + ', ' + str(a[2])
          + ', ' + str(a[3]) + ')')
#plt.savefig('distMatrix' + str(a[0]) + '.jpg', bbox_inches='tight')
plt.show()