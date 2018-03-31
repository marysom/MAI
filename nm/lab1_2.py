from numpy import *
import numpy as np
f = open('input_lab1_2.txt')
line = f.readlines()
a=line[0].split()
a=list(map(int, a))
b=line[1].split()
b=list(map(int, b))
c=line[2].split()
c=list(map(int, c))
d=line[3].split()
d=list(map(int, d))
#print('a = ',a)
#print('b = ',b)
#print('c = ',c)
#print('d = ',d)
f.close()
print('Лабораторная работа 1.2')
print('Метод прогонки')
print('Вариант 14')
def prog(a,b,c,d):
    n=len(d)
    x=np.zeros(n)
    p=np.zeros(n)
    q=np.zeros(n)
    i=0
    p[0]=-c[0]/b[0]
    q[0]=d[0]/b[0]
    for i in range(1,n-1):
        p[i]=-c[i]/(a[i]*p[i-1]+b[i])
    print('p = ',p)
    i=0
    for i in range(1,n):
        q[i]=(d[i]-a[i]*q[i-1])/(a[i]*p[i-1]+b[i])
    print('q = ',q)
    i=3
    x[n-1]=q[n-1]
    while i>=0:
        x[i]=p[i]*x[i+1]+q[i]
        i=i-1
    return x
print('Решение СЛАУ: x = ',prog(a,b,c,d))
