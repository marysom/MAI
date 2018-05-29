import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from sympy import Symbol
import math
def inputdata():
    f = open('input_lab3_4.txt')
    line = f.readlines()
    count=len(line)
    x=line[0].split()
    x=list(map(float,x))
    y=line[1].split()
    y=list(map(float,y))
    x_test=float(line[2])
    f.close()
    res = (x,y,x_test)
    return res

def print_table(x,y):
    n = len(x)
    print('-------------------------------------------------------------------')
    print('| i        ',end='')
    print('| ',0,'      ',end='')
    for i in range(1,n):
        print('|',i,'       ',end='')
    print('|')
    print('-------------------------------------------------------------------')
    print('| x[i]     ',end='')
    for i in range(n):
        print('| ',"%.4f" %x[i],' ',end='')    
    print('|')
    print('-------------------------------------------------------------------')
    print('| y[i]     ',end='')
    for i in range(n):
        print('| ',"%.4f" %y[i],' ',end='')
    print('|')
    print('-------------------------------------------------------------------')
    return


def f2(x,y):
    f2x = np.zeros(len(x) - 1)
    i = 0
    for j in range(1,len(x)):
        f2x[i] = (y[i]- y[j])/(x[i] - x[j])
        i += 1
        if i == len(x):
            break
    return f2x

def f3(x,y):
    f3x = np.zeros(len(x) - 2)
    i = 0
    for k in range(2,len(x)):
        fl = f2(x,y)
        f3x[i] = (fl[i] - fl[i+1])/(x[i] - x[k])
        i += 1
        if i == len(x) - 1:
            break
    return f3x

def f4(x,y):
    f4x = np.zeros(len(x) - 3)
    i = 0
    for k in range(3,len(x)):
        fl = f3(x,y)
        f4x[i] = (fl[i] - fl[i+1])/(x[i] - x[k])
        i += 1
        if i == len(x) - 2:
            break
    return f4x

def f5(x,y):
    f5x = np.zeros(len(x) - 4)
    i = 0
    for k in range(4,len(x)):
        fl = f4(x,y)
        f5x[i] = (fl[i] - fl[i+1])/(x[i] - x[k])
        i += 1
        if i== len(x) - 3:
            break
    return f5x
     

def tablraz(x,y):
    ff2 = f2(x,y)
    ff3 = f3(x,y)
    ff4 = f4(x,y)
    ff5 = f5(x,y)
    print('Таблица разделенных разностей')
    n = len(x)
    print('-------------------------------------------------------------------')
    print('|x[i]      ',end='')
    print('  ',"%.4f" %x[0],'    ',end='')
    for i in range(1,n):
        print('',"%.4f" %x[i],'  ',end='')
    print(' |')
    print('-------------------------------------------------------------------')
    print('|f(x[i])   ',end='')
    print('  ',"%.4f" %y[0],'    ',end='')
    for i in range(1,n):
        print('',"%.4f" %y[i],'  ',end='')
    print(' |')
    print('-------------------------------------------------------------------')
    print('|f(x[i]x[j])',end='')
    print('     ',"%.4f" %ff2[0],'  ',end='')
    for i in range(1,len(ff2)):
        print(' ',"%.4f" %ff2[i],'  ',end='')
    print('      |')
    print('-------------------------------------------------------------------')
    print('|f(x[i]x[j]x[k])',end='')
    print('     ',"%.4f" %ff3[0],'  ',end='')
    for i in range(1,len(ff3)):
        print(' ',"%.4f" %ff3[i],'  ',end='')
    print('          |')
    print('-------------------------------------------------------------------')
    print('|f(x[i]x[j]x[k]x[l])',end='')
    print('        ',"%.4f" %ff4[0],'  ',end='')
    for i in range(1,len(ff4)):
        print(' ',"%.4f" %ff4[i],'  ',end='')
    print('                 |')
    print('-------------------------------------------------------------------')
    print('|f(x[i]x[j]x[k]x[l]x[m])',end='')
    print('          ',"%.4f" %ff5[0],'  ',end='')
    print('                     |')
    print('-------------------------------------------------------------------')
    return


def phi1(x,y,x_test):
    ff2 = f2(x,y)
    ff3 = f3(x,y)
    ff4 = f4(x,y)
    ff5 = f5(x,y)
    sum1 = ff3[0] * (2 * x_test - x[0] - x[1])
    sum2 = ff4[0] * (3 * math.pow(x_test,2) - 2 * x_test *(x[0]+x[1]+x[2])+x[1]*x[2]+x[0]*x[2]+x[0]*x[1])
    sum3 = ff5[0] * (4 * math.pow(x_test,3) - 3 * math.pow(x_test,2)*(x[0]+x[1]+x[2]+x[3])+2*x_test*(x[2]*x[3]+x[1]*x[3]+x[0]*x[3]+x[1]*x[2]+x[0]*x[2]+x[0]*x[1])-x[0]*x[1]*x[2]-x[0]*x[1]*x[3]-x[0]*x[2]*x[3]-x[1]*x[2]*x[3])
    return ff2[0] + sum1 + sum2 + sum3


def phi2(x,y,x_test):
    ff2 = f2(x,y)
    ff3 = f3(x,y)
    ff4 = f4(x,y)
    ff5 = f5(x,y)
    sum1 = ff4[0]*(6 * x_test - 2*(x[0]+x[1]+x[2]))
    sum2 = ff5[0]*(12 * math.pow(x_test,2) - 6 * x_test *(x[0]+x[1]+x[2]+x[3]) + 2*(x[2]*x[3]+x[1]*x[3]+x[0]*x[3]+x[1]*x[2]+x[0]*x[2]+x[0]*x[1]))
    return 2*ff3[0] + sum1 + sum2

def plotf(x,y,x_test):
    plt.figure()
    plt.grid()
    for i in range(len(x)):
        plt.scatter(x[i],y[i],color = 'green', s = 15)        
    plt.scatter(x_test, 0, color = 'red', s = 15)
    plt.xlim([0,8])
    plt.ylim([0,8])
    plt.show()
    return

def main():
    
    print('Лабораторная работа 3.4')
    print('Вариант 14')
    inp = inputdata()
    (x,y,x_test) = inp
    n = len(x)
    print_table(x,y)
    print('X* = ',x_test)
    tablraz(x,y)
    print("f'(X*) = ",phi1(x,y,x_test))
    print('f"(X*) = ',phi2(x,y,x_test))
    plotf(x,y,x_test)
if __name__ == "__main__":
    main()
