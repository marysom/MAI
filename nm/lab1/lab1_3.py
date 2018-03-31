from numpy import *
import numpy as np
import math 
f = open('input_lab1_3.txt')
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
print('Лабораторная работа 1.3')
print('Вариант 14')
f.close()
print('А=',a)
print('b=',b)

def norma_matrix(a,count):
    norm=0.
    for i in range(0,count-1):
        sum_a=0
        
        for j in range(0,count-1):
            sum_a+=math.fabs(a[i,j])
        if sum_a>norm:
            norm=sum_a
    return norm

def norma_vector(b):
    normv=b[0]
    for i in range(1,len(b)-1):
        if math.fabs(b[i])>normv:
            normv=math.fabs(b[i])
    return normv

def method_yakobi(a,b,count):
    alfa=np.zeros((count,count))
    beta=np.zeros(count)
    for i in range(0,count):
        beta[i]=b[i]/a[i,i]
        for j in range(0,count):
            if i==j:
                alfa[i,j]=0
            else:
                alfa[i,j]=-a[i,j]/a[i,i]
    res=(alfa,beta)
    return res

def matr_vect(a,b):
    res=np.zeros(len(b))
    for i in range(0,len(b)):
        for j in range(0,len(b)):
            res[i]+=a[i,j]*b[j]
    return res

def vect_p_vect(b,c):
    res=np.zeros(len(b))
    for i in range(0,len(b)):
        res[i]=b[i]+c[i]
    return res

def vect_m_vect(b,c):
    res=np.zeros(len(b))
    for i in range(0,len(b)):
        res[i]=b[i]-c[i]
    return res

def sum_razl(a,count):
    b=np.zeros((count,count))
    c=np.zeros((count,count))
    for i in range(0,count):
        for j in range(0,count):
            if (j<=i):
                b[i,j]=a[i,j]
            if (j>i):
                c[i,j]=a[i,j]
    res=(b,c)
    return res

def matr_m_matr(a,b,count):
    for i in range(0,count-1):
        for j in range(0,count-1):
            a[i,j]-=b[i,j]
    return a

'''+++++++++++++++++++++++++++++++++++++++'''
def mpi(a,b,count,eps):
    result=method_yakobi(a,b,count-1)
    (alfa,beta)=result
    print('alfa=',alfa)
    print('beta=',beta)
    print('--------------------------------------------')
    if norma_matrix(alfa,count)<1:
        print('Условие выполнено:')
        print('||alfa||=',norma_matrix(alfa,count),'<1')
    print('--------------------------------------------')
    q=(norma_matrix(alfa,count)/(1-norma_matrix(alfa,count)))
    x=beta
    k=0
    while True:
      tmp=x
      x=matr_vect(alfa,tmp)
      x=vect_p_vect(beta,x)
      epsk=vect_m_vect(x,tmp)
      epsk=q*norma_vector(epsk)
      k+=1
      print(k,' итерация')
      print(x)
      if epsk<eps:
        break
    return

def zeidel(a,b,count,eps):
    result=method_yakobi(a,b,count-1)
    (alfa,beta)=result
    print('alfa=',alfa)
    print('beta=',beta)
    print('--------------------------------------------')
    if norma_matrix(alfa,count)<1:
        print('Условие выполнено:')
        print('||alfa||=',norma_matrix(alfa,count),'<1')
    print('--------------------------------------------')
    q=(norma_matrix(alfa,count)/(1-norma_matrix(alfa,count)))
    x=beta
    k=0
    while True:
        tmp=x
        res=sum_razl(alfa,count-1)
        (bb,cc)=res
        ee=np.eye(count-1)
        dd=matr_m_matr(ee,bb,count)
        dd=linalg.inv(dd)
        x=vect_p_vect(matr_vect(dd,matr_vect(cc,x)),matr_vect(dd,beta))
        epsk=vect_m_vect(x,tmp)
        epsk=norma_vector(epsk)
        epsk=(norma_matrix(cc,count)/(1-norma_matrix(alfa,count)))*epsk
        k+=1
        print(k,' итерация')
        print(x)
        if epsk<eps:
            break
    return

while True:
    print('--------------------------------------------')
    print('Выберите метод решения СЛАУ:')
    print('1 - метод простых итераций')
    print('2 - метод Зейделя')
    print('0 - выход')
    method=int(input())
    if method==1:
        print('Введите точность вычислений:')
        eps=float(input())
        print('eps=',eps)
        print('--------------------------------------------')
        mpi(a,b,count,eps)
    if method==2:
        print('Введите точность вычислений:')
        eps=float(input())
        print('eps=',eps)
        print('--------------------------------------------')
        zeidel(a,b,count,eps)
    if (method!=0) and (method!=1) and (method!=2):
        print('Упс! Такого метода не существует. Попытайтесь снова.')
    if method==0:
        break
    
