from sympy import diff,symbols
import matplotlib.pyplot as plt
import numpy as np
import math

def f(x):
    return x**3 - 2*(x**2) - 10*x + 15
def f1(x):
    return 3*x**2 - 4*x - 10
def f2(x):
    return 6*x - 4
def phi(x):
    return 0.1*x**3 - 0.2*x**2 + 1.5
def phi1(x):
    return 0.3*x**2 - 0.4*x


def plots():
    plt.figure()
    x=np.arange(-7.,7.,0.001)
    plt.plot(x,x,color='red')
    plt.plot(x,0.1*(x**3)-0.2*(x**2)+1.5,color='blue')
    plt.grid()
    plt.xlim(-7.,7.)
    plt.ylim(-5.,5.)
    plt.show()
    return

def start(a,b):
    x0=a
    while (f(x0)*f2(x0)<=0) and (x0<b):
        x0+=0.0001
    return x0

def newton(a,b,eps):
    k=0
    x_old=start(a,b)
    while True:
        k+=1
        x_new=x_old-(f(x_old)/f1(x_old))
        print('================')
        print(k, 'итерация')
        print('x = ',x_new)
        epsk=math.fabs(x_new-x_old)
        print('epsk = ',epsk)
        x_old=x_new
        if epsk < eps:
            break
    return x_new

def calc_q(a,b):
    q = math.fabs(phi1(a))
    h=a+0.001
    while h<b:
        qq=math.fabs(phi1(h))
        if qq>q:
            q=qq
        h+=0.001
    return q

def mpi(a,b,eps):
    k=0
    x0=a
    x_old=x0
    q = calc_q(a,b)
    print('q = ',q)
    while True:
        k+=1
        x_new = phi(x_old)
        print('================')
        print(k, 'итерация')
        print('x = ',x_new)
        epsk=math.fabs(x_new-x_old)*q/(1-q)
        print('epsk = ',epsk)
        x_old=x_new
        if epsk < eps:
            break
    return x_new
    return 

def main():
    print('Лабораторная работа 2.1\nВариант 14')
    x, y = symbols('x, y')
    ff=f(x)
    ff1=f(x).diff(x)
    ff2=f(x).diff(x).diff(x)
    print("f(x) = ",ff,"\nf'(x) = ",ff1,"\nf''(x) = ",ff2)
    plots()
    print('Функция имеет 2 положительных корня на отрезках [0.,2.], [2.,4.] и\nотрицательный корень на отрезке [-4.,-2.]')
    while True:
        print('\nВыберите метод:\n1 - Метод Ньютона\n2 - Метод простых итераций\n0 - выход')
        k=int(input())
        if k==0:
            break
        if k==1:
            print('\nМетод Ньютона')
            print('Введите точность вычислений:')
            eps=float(input())
            print('Введите отезок:')
            a=float(input())
            b=float(input())
            newton(a,b,eps)
        if k==2:
            print('\nМетод простых итераций')
            print('Введите точность вычислений:')
            eps=float(input())
            print('Введите отезок:')
            a=float(input())
            b=float(input())
            mpi(a,b,eps)
    return
    
main()
