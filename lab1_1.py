from numpy import *
import numpy as np
import math 
f = open('input_lab1_1.txt')
line = f.readlines()
count=len(line)
a=line[0].split()
a=list(map(float,a))
for i in range(1,count-1):
    tmp=line[i].split()
    tmp=list(map(float,tmp))
    a=np.row_stack((a,tmp))
b=line[count-1].split()
b=list(map(float,b))
print('Лабораторная работа 1.1')
print('Вариант 14')
f.close()
print('А=',a)
print('b=',b)
print('-------------------------------------')
print('Найдем решение СЛАУ с помощью функции linalg.solve(x,y):')
print('x=',linalg.solve(a,b))
print('-------------------------------------')
def LU(a,b):
    u=a
    l=np.zeros([len(b),len(b)])
    for k in range(1,len(b)):
        for i in range(k-1,len(b)):
            for j in range(i,len(b)):
                l[j,i]=u[j,i]/u[i,i]
        for i in range(k,len(b)):
            for j in range(k-1,len(b)):
                u[i,j]=u[i,j]-l[i,k-1]*u[k-1,j]
    result=(u,l)
    return result
res=LU(a,b)
(u,l)=res
print('U=',u)
print('L=',l)
print('Проверка:')
print('L*U=',l@u)        
z=np.zeros(len(b))
for i in range(0,len(b)):
    sum=0
    for j in range(0,i):
        sum+=l[i,j]*z[j]
    z[i]=b[i]-sum
print('LUx=b')
print('Lz=b')
print('z=',z)
x=np.zeros(len(b))
for i in range(len(b)-1,-1,-1):
    sum=0
    for j in range(i+1,len(b)):
        sum+=u[i,j]*x[j]
    x[i]=(1/u[i,i])*(z[i]-sum)
print('Ux=z')
print('x=',x)
