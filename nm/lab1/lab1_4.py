from numpy import *
import numpy as np
import math 
f = open('input_lab1_4.txt')
line = f.readlines()
count=len(line)
a=line[0].split()
a=list(map(float,a))
for i in range(1,count):
    tmp=line[i].split()
    tmp=list(map(float,tmp))
    a=np.row_stack((a,tmp))
print('Лабораторная работа 1.4')
print('Метод вращений')
print('Вариант 14')
f.close()
print('А=',a)
print('Введите точность вычислений:')
eps=float(input())
print('--------------------------------------------')
def maxel(a,count):
    maxelem=math.fabs(a[0,1])
    l=0
    m=1
    for i in range(0,count):
        for j in range(i+1,count):
            if (math.fabs(a[i,j])>maxelem) and (i<j):
                maxelem=math.fabs(a[i,j])
                l=i
                m=j
    return maxelem,l,m

def multi_2matr(a,b,count):
    c=np.zeros((count,count))
    for i in range(0,count):
        for j in range(0,count):
            for k in range(0,count):
                c[i,j]+=a[i,k]*b[k,j]
    return c

def off(a,l,m,count):
    summ=0
    for l in range(0,count):
        for m in range(l+1,count):
            summ=summ+(a[l,m]*a[l,m])
    return math.sqrt(summ)

def trans(a,count):
    result=maxel(a,count)
    (maxelem,l,m)=result
    if a[l,l]==a[m,m]:
        phi=math.pi/4
    else:
        phi=0.5*(math.atan(2*a[l,m]/(a[l,l]-a[m,m])))
    print('max=',maxelem,' l=',l,' m=',m)
    print('phi=', phi)
    c=math.cos(phi)
    s=math.sin(phi)
    u=np.eye(count)
    u[l,l]=c
    u[l,m]=-s
    u[m,m]=c
    u[m,l]=s
    ut=np.eye(count)
    ut[l,l]=c
    ut[m,m]=c
    ut[l,m]=s
    ut[m,l]=-s
    res=(u,ut,l,m)
    return res
def rotation(a,count,eps):
    k=0
    v=np.eye(count)
    while True:
        result=trans(a,count)
        (u,ut,l,m)=result
        a=multi_2matr(multi_2matr(ut,a,count),u,count)
        v=multi_2matr(v,u,count)
        print('++++++++++++++++++++++++++++++++++++')
        k+=1
        print(k, 'итерация')
        print('u=',u)
        print('ut=',ut)
        print('a=',a)
        print('v=',v)
        res=(a,v)
        if off(a,l,m,count)<eps:
            break
    return res
result=rotation(a,count,eps)
(a,v)=result
print('--------------------------------------------')
print('Собственные значения:')
for i in range(0,count):
    print(a[i,i])
print('--------------------------------------------')
print('Собственные векторы:')
for i in range(0,count):
    sz=np.zeros(count)
    for j in range(0,count):
        sz[j]=v[j,i]
    print(sz)
    
  
