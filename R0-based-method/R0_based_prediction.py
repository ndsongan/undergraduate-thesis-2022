#!/usr/bin/env python3
# -*- coding: utf-8 -*-



from random import random, randint
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import datetime

xls = pd.ExcelFile('ThuDucData.xlsx')
df1 = pd.read_excel(xls, 'Sheet1')
df = pd.read_excel(xls, 'Sheet2')
#df = pd.read_excel('ThuDucBook.xlsx')

n = df['Name']

month = 7
day = 25

# using data from previous day to predict current day
# F0 is prediction for day + 1
F0 = list(df[datetime.datetime(2021, month, day)])

P = [23154, 7548, 19956, 11871, 8983, 11973, 6526, 25943, 17139, 18925, 15261, 
     20327, 13863, 21890, 18467, 133206, 6700, 11580, 5576, 4430, 16714, 54160, 
     14884, 11983, 14973, 12468, 15640, 8638, 8355, 8559, 18754]
D = [5907, 737, 3635, 10992, 2604, 5393, 975, 4144, 2261, 8563, 11739, 6891, 
     9832, 3210, 4822, 3806, 285, 891, 457, 357, 18993, 81200, 2797, 3514, 
     5024, 2587, 3460, 1941, 651, 2282, 4585]

R0 = [0 for a in range(len(D))]
X = [0 for a in range(len(D))]

N = len(D)
Days = 2
minn = min(D)
maxx = max(D)
R0_min = 0.5
R0_max = 2.0

for i in range(N):
    R0[i] = R0_min + ((D[i] - minn) / (maxx - minn)) * (R0_max - R0_min)
    X[i] = int(round(P[i] / 1000))

def simulation(F0, Days):
    F = []
    for d in range(Days):
        for w in range(N):
            #each day infected individuals will increase about (4xR0)%
            F0[w] += 0.05*F0[w] * R0[w]
            c = 0
            for p in range(X[w]):
                if random() <= 0.001:
                    R = 1.0
                else:
                    R = R0[w]
                #generate the path of people p
                visit_nodes = randint(1,5)
                path = [randint(0,20) for a in range(visit_nodes)]
            
                for v in path:
                    if R > R0[v]:
                        #an individual has very little influence on where they come
                        R0[v] = R0[v] + abs(R-R0[v])*0.001
                    else:
                        # but individuals are affected (about 10 times) when coming 
                        # to high-risk areas
                        R = R + abs(R-R0[v])*0.01
                c += R
            c = c/X[w]
            #an individual has very little influence on where they come
            R0[w] = R0[w] + (c - R0[w])*0.001
            for m in range(len(F0)):
                F0[m]=round(F0[m])
        temp = [a for a in F0]
        F.append(temp)
    return F

F = simulation(F0, Days)



def visualize_per_day(real, predict):
    
    x = [a for a in real]
    
    X = np.arange(len(n))  # the label locations
    width = 0.28  # the width of the bars
    
    fig, ax = plt.subplots(figsize=(8, 8), dpi=80)
    ax.bar(X - width, x, width, label= 'real ' + 
           str(datetime.datetime(2021, month, day + 1)))
    rects2 = ax.bar(X, predict, width, label='predict ' +
                    str(datetime.datetime(2021, month, day + 1)) )
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylim(0, 2000)
    ax.set_ylabel('Infected')
    ax.set_xlabel('Districts')
    ax.set_title('Thu Duc COVID-19 for ' + str(datetime.datetime(2021, 8, 12)))
    ax.set_xticks(X)
    ax.set_xticklabels(n)
    plt.xticks(rotation=80)
    ax.legend()
    
    fig.tight_layout()
    
    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%d' % int(height),
                    ha='center', va='bottom', rotation = 80)
    
    autolabel(rects2)
    # plt.savefig('july' + str(date) + '_' + str(date + 1) + '_' + str(date + 2) 
    #             + '.pdf')
    plt.show()

visualize_per_day(df[datetime.datetime(2021, month, day)], F[0])


