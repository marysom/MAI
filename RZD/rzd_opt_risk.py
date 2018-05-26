from numpy import *
from datetime import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.lines

matplotlib.rcParams["lines.linewidth"] = 1
matplotlib.rcParams["lines.linestyle"] = "--"

data = pd.read_csv("newdata1.csv", ';')
data.columns=['path','year','days']
print(data.head())
print(data.info())
print(data.isnull().any()) 

path=data['path'].values

data1=data.dropna()
print('Данные без NA (отсортированные)')
data1=data1.sort_values('path', ascending=True)
print(data1.head())

a=len(data1['path'])
a_yel=math.floor(a/4)
a_or=math.floor(3*a/4)
a_red=math.floor(a)
path1=data1['path'].values
days1=data1['days'].values
print('Количество сходов: ',a)
print('В зеленой зоне =  0')
print('В желтой зоне >= ',a_yel)
print('В оранжевой зоне >= ',a_or)
print('В красной зоне = ',a_red)

def countt(path1,days1,a):
    count = 0
    for i in range(len(path1)):
        if (path1[i] <= path1[a]) and (days1[i] <= days1[a]):
            count += 1
    return count

def opt1(path1,days1,a):
    n = a
    k = countt(path1,days1,a)
    while k < a and n < len(path1):
        n += 1
        k = countt(path1,days1,n)
    return n

n1_yel = opt1(path1,days1,a_yel)
print('________________')
n1_or = opt1(path1,days1,a_or)
print('В зеленой зоне: ', 0)
#print('n1_yel = ',n1_yel)
print('В желтой зоне: ', countt(path1,days1,n1_yel))
#print('n1_or = ',n1_or)
print('В оранжевой зоне: ', countt(path1,days1,n1_or))
print('В красной зоне = ', a)
print('-----------------')


plt.figure()
plt.scatter(data1['path'], data1['days'], color='black', s=5)

plt.plot([0,path1[0]-5],[days1[0]-10,days1[0]-10],color = 'green')
plt.plot([path1[0]-5,path1[0]-5],[0,days1[0]-10],color = 'green')

plt.plot([0,path1[n1_yel]],[days1[n1_yel],days1[n1_yel]],color = 'yellow')
plt.plot([path1[n1_yel],path1[n1_yel]],[0,days1[n1_yel]],color = 'yellow')

plt.plot([0,path1[n1_or]],[days1[n1_or],days1[n1_or]],color = 'orange')
plt.plot([path1[n1_or],path1[n1_or]],[0,days1[n1_or]],color = 'orange')

plt.plot([0.,max(path1)],[max(days1),max(days1)],color='red')
plt.plot([max(path1),max(path1)],[0.,max(days1)],color='red')

plt.show()


