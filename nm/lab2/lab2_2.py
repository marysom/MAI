from sympy import diff,symbols
import matplotlib.pyplot as plt
import numpy as np
import math
from itertools import product
from numpy.linalg import norm, solve, det

def g1(x):
    return math.sqrt(9-(x**2))/2
def g2(x):
    return (x + math.exp(x))/3


def f1(x1,x2):
    return (x1**2)/9 + 4*(x2**2)/9 - 1
def f2(x1,x2):
    return 3*x2 - math.exp(x1) - x1
def f11(x1,x2):
    return 2*x1/9
def f12(x1,x2):
    return 8*x2/9
def f21(x1,x2):
    return -math.exp(x1) - 1
def f22(x1,x2):
    return 3
def jf(x1,x2):
    return np.array([[f11(x1,x2),f12(x1,x2)],[f21(x1,x2),f22(x1,x2)]])

def phi1(x1, x2):
    lmbd = calc_lambda()
    return x1 - (f1(x1, x2) * lmbd[0, 0] + f2(x1, x2) * lmbd[0, 1])
def phi2(x1, x2):
    lmbd = calc_lambda()
    return x2 - (f1(x1, x2) * lmbd[1, 0] + f2(x1, x2) * lmbd[1, 1])
def phi11(x1, x2):
    lmbd = calc_lambda()
    return 1 - (f11(x1, x2) * lmbd[0, 0] + f21(x1, x2) * lmbd[0, 1])
def phi12(x1, x2):
    lmbd = calc_lambda()
    return -(f12(x1, x2) * lmbd[0, 0] + f22(x1, x2) * lmbd[0, 1])
def phi21(x1, x2):
    lmbd = calc_lambda()
    return -(f11(x1, x2) * lmbd[1, 0] + f21(x1, x2) * lmbd[1, 1])
def phi22(x1, x2):
    lmbd = calc_lambda()
    return 1 - (f12(x1, x2) * lmbd[1, 0] + f22(x1, x2) * lmbd[1, 1])
def jphi(x):
    return np.array([[phi11(*x), phi12(*x)],[phi21(*x), phi22(*x)]])

def a1(x1,x2):
    return np.array([[f1(x1,x2),f12(x1,x2)],[f2(x1,x2),f22(x1,x2)]])
def a2(x1,x2):
    return np.array([[f11(x1,x2),f1(x1,x2)],[f21(x1,x2),f2(x1,x2)]])

def plots():
    plt.figure()
    x = np.linspace(-3, 3, 10000)
    y11=[g1(i) for i in x]
    y12=[-g1(i) for i in x]
    y2=[g2(i) for i in x]
    plt.plot(x,y11,color='red')
    plt.plot(x,y12,color='red')
    plt.plot(x,y2,color='blue')
    plt.xlim(-4.,4.)
    plt.ylim(-3.,5.)
    plt.grid()
    plt.show()
    return

def newton(eps):
    x0=[1,g2(1)]
    k=0
    x_old=x0
    while True:
        x_new=np.zeros(2)
        A11 = a1(x_old[0],x_old[1])
        A22 = a2(x_old[0],x_old[1])
        J11 = jf(x_old[0],x_old[1])
        deta1=det(A11)
        deta2=det(A22)
        detj=det(J11)
        x_new[0]=x_old[0] - (deta1/detj)
        x_new[1]=x_old[1] - (deta2/detj)
        k+=1
        print(k, 'итерация')
        print('x = ',x_new)
        epsk=norm(x_new-x_old)
        print('epsk = ',epsk,'\n')
        x_old=x_new
        if epsk<eps:
            break
    return x_new

def calc_lambda():
    shape = 2
    current_j = jf(1., 1.)
    inv_j = np.array([solve(current_j, i) for i in np.eye(shape)])
    return np.transpose(inv_j)

def calc_q():
    x1 = np.linspace(1., 1.4, 100)
    x2 = np.linspace(1., 1.4, 100)
    points = list(product(x1, x2))
    vals = [norm(jphi(point), np.inf) for point in points]
    q = np.max(vals)
    return q

def mpi(eps):
    x0=[1,1]
    k=0
    x_old=x0
    q=calc_q()
    while True:
        x_new=np.array([phi1(*x_old),phi2(*x_old)])
        k+=1
        print(k, 'итерация')
        print('x = ',x_new)
        print(q)
        epsk = norm(x_new - x_old) * q / math.fabs(1-q)
        print('epsk = ',epsk,'\n')
        x_old=x_new
        if epsk<eps:
            break        
    return x_new

def main():
    print('Лабораторная работа 2.2\nВариант 14')
    print('(x1/3)^2 + (2*x2/3)^2 - 1 = 0\n3*x2 - e^x1 - x1 = 0')
    print('Введите точность вычислений:')
    eps = float(input())
    print('-------------------------------------------')
    print('Метод простых итераций')
    print('lambda = ', calc_lambda())
    print('q = ',calc_q())
    mpi(eps)
    print('-------------------------------------------')
    print('Метод Ньютона')
    newton(eps)
    plots()
if __name__ == "__main__":
    main()
