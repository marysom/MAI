import numpy as np
import math
import matplotlib.pyplot as plt

def eiler(x,h):
    y = np.zeros(len(x))
    y[0] = 1
    z = np.zeros(len(x))
    z[0] = 3
    dy = np.zeros(len(x) - 1)
    dz = np.zeros(len(x) - 1)
    for i in range(len(x)-1):
        y[i+1] = y[i] + h * f(x[i],y[i],z[i])
        z[i+1] = z[i] + h * g(x[i],y[i],z[i])
        dy[i] = h * f(x[i],y[i],z[i])
        dz[i] = h * g(x[i],y[i],z[i])
    return y, z, dy, dz
    

def runge_kuttt(x,h,n):
    y = np.zeros(len(x))
    y[0] = 1
    z = np.zeros(len(x))
    z[0] = 3
    dy = np.zeros(len(x) - 1)
    dz = np.zeros(len(x) - 1)
    for i in range(n):
        K1 = h * f(x[i],y[i],z[i])
        L1 = h * g(x[i],y[i],z[i])
        K2 = h * f(x[i] + 0.5*h,y[i] + 0.5*K1,z[i] + 0.5*L1)
        L2 = h * g(x[i] + 0.5*h,y[i] + 0.5*K1,z[i] + 0.5*L1)
        K3 = h * f(x[i] + 0.5*h,y[i] + 0.5*K2,z[i] + 0.5*L2)
        L3 = h * g(x[i] + 0.5*h,y[i] + 0.5*K2,z[i] + 0.5*L2)
        K4 = h * f(x[i] + h,y[i] + K3,z[i] + L3)
        L4 = h * g(x[i] + h,y[i] + K3,z[i] + L3)

        if i < n - 1:
            dy[i] = (K1 + 2*K2 + 2*K3 + K4)/6
            dz[i] = (L1 + 2*L2 + 2*L3 + L4)/6
            y[i+1] = y[i] + dy[i]
            z[i+1] = z[i] + dz[i]
    return y, z, dy, dz

def f(x,y,z):
    return z

def adams(x,h):
    y, z, dy, dz = runge_kuttt(x,h,4)
    for i in range(3,len(x)-1):
        y[i+1] = y[i] + h*(55 * z[i] - 59 * z[i-1] + 37 * z[i-2] - 9 * z[i-3])/24
        z[i+1] = z[i] + h*(55 * g(x[i],y[i],z[i]) - 59 * g(x[i-1],y[i-1],z[i-1]) + 37 * g(x[i-2],y[i-2],z[i-2] - 9 * g(x[i-3],y[i-3],z[i-3])))/24
    return y, z

def g(x,y,z):
    return 2*x*z/(1+x**2)
def true_val(x):
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = x[i]**3 + 3* x[i] + 1
    return y

def print_eiler(x,y,z,dy,dz,tv):
    n = len(x)
    print('Метод Эйлера')
    print('------------------------------------------------------------------------------------------')
    print('| i        ',end='')
    print('| ',0,'        ',end='')
    for i in range(1,n):
        print('|',i,'         ',end='')
    print('|')
    print('------------------------------------------------------------------------------------------')
    print('| x[i]     ',end='')
    for i in range(n):
        print('| ',"%.6f" %x[i],' ',end='')    
    print('|')
    print('------------------------------------------------------------------------------------------')
    print('| y[i]     ',end='')
    for i in range(n):
        print('| ',"%.6f" %y[i],' ',end='')
    print('|')
    print('------------------------------------------------------------------------------------------')
    print('| z[i]     ',end='')
    for i in range(n):
        print('| ',"%.6f" %z[i],' ',end='')
    print('|')
    print('------------------------------------------------------------------------------------------')

    print('|  dy[i]   ',end='')
    for i in range(n-1):
        print('| ',"%.6f" %dy[i],' ',end='')
    print('|')
    print('------------------------------------------------------------------------------------------')
    print('|  dz[i]   ',end='')
    for i in range(n-1):
        print('| ',"%.6f" %dz[i],' ',end='')
    print('|')
    print('------------------------------------------------------------------------------------------')

    print('|y_true[i] ',end='')
    for i in range(n):
        print('| ',"%.6f" %tv[i],' ',end='')
    print('|')
    print('------------------------------------------------------------------------------------------')
    print('|   eps[i] ',end='')
    for i in range(n):
        print('| ',"%.2e" %math.fabs(tv[i] - y[i]),' ',end='')
    print('|')
    print('------------------------------------------------------------------------------------------')
    return



def print_runge(x,y,z,dy,dz,tv):
    n = len(x)
    print('Метод Рунге-Кутты')
    print('------------------------------------------------------------------------------------------')
    print('| i        ',end='')
    print('| ',0,'        ',end='')
    for i in range(1,n):
        print('|',i,'         ',end='')
    print('|')
    print('------------------------------------------------------------------------------------------')
    print('| x[i]     ',end='')
    for i in range(n):
        print('| ',"%.6f" %x[i],' ',end='')    
    print('|')
    print('------------------------------------------------------------------------------------------')
    print('| y[i]     ',end='')
    for i in range(n):
        print('| ',"%.6f" %y[i],' ',end='')
    print('|')
    print('------------------------------------------------------------------------------------------')
    print('| z[i]     ',end='')
    for i in range(n):
        print('| ',"%.6f" %z[i],' ',end='')
    print('|')
    print('------------------------------------------------------------------------------------------')

    print('|  dy[i]   ',end='')
    for i in range(n-1):
        print('| ',"%.6f" %dy[i],' ',end='')
    print('|')
    print('------------------------------------------------------------------------------------------')
    print('|  dz[i]   ',end='')
    for i in range(n-1):
        print('| ',"%.6f" %dz[i],' ',end='')
    print('|')
    print('------------------------------------------------------------------------------------------')

    print('|y_true[i] ',end='')
    for i in range(n):
        print('| ',"%.6f" %tv[i],' ',end='')
    print('|')
    print('------------------------------------------------------------------------------------------')
    print('|   eps[i] ',end='')
    for i in range(n):
        print('| ',"%.2e" %math.fabs(tv[i] - y[i]),' ',end='')
    print('|')
    print('------------------------------------------------------------------------------------------')
    return

def print_adams(x,y,z,tv):
    n = len(x)
    print('Метод Адамса')
    print('------------------------------------------------------------------------------------------')
    print('| i        ',end='')
    print('| ',0,'        ',end='')
    for i in range(1,n):
        print('|',i,'         ',end='')
    print('|')
    print('------------------------------------------------------------------------------------------')
    print('| x[i]     ',end='')
    for i in range(n):
        print('| ',"%.6f" %x[i],' ',end='')    
    print('|')
    print('------------------------------------------------------------------------------------------')
    print('| y[i]     ',end='')
    for i in range(n):
        print('| ',"%.6f" %y[i],' ',end='')
    print('|')
    print('------------------------------------------------------------------------------------------')
    print('| z[i]     ',end='')
    for i in range(n):
        print('| ',"%.6f" %z[i],' ',end='')
    print('|')
    print('------------------------------------------------------------------------------------------')
    print('|y_true[i] ',end='')
    for i in range(n):
        print('| ',"%.6f" %tv[i],' ',end='')
    print('|')
    print('------------------------------------------------------------------------------------------')
    print('|   eps[i] ',end='')
    for i in range(n):
        print('| ',"%.2e" %math.fabs(tv[i] - y[i]),' ',end='')
    print('|')
    print('------------------------------------------------------------------------------------------')
    return
def runger(x,h,flag):
    y1 = true_val(x)
    if flag == 1:
        #y1, z1, dy1, dz1 = eiler(x,h)
        y2, z2, dy2, dz2 = eiler(x,2*h)
        eps = np.zeros(len(x))
        for i in range(len(x)):
            eps[i] = (y2[i] - y1[i])/(2**1 - 1)
    if flag == 2:
        n = len(x) 
        #y1, z1, dy1, dz1 = runge_kuttt(x,h,n)
        y2, z2, dy2, dz2 = runge_kuttt(x,2*h,n)
        eps = np.zeros(len(x))
        for i in range(len(x)):
            eps[i] = (y2[i] - y1[i])/(2**4 - 1)
    if flag == 3:
        #y1, z1 = adams(x,h)
        y2, z2 = adams(x,2*h)
        eps = np.zeros(len(x))
        for i in range(len(x)):
            eps[i] = (y2[i] - y1[i])/(2**4 - 1)
    return eps

def plots(x,h):
    n = len(x)
    xtv = np.linspace(0.,1.,1000)
    tv = true_val(xtv)
    y1, z1, dy1, dz1 = eiler(x,h)
    y2, z2, dy2, dz2 = runge_kuttt(x,h,n)
    y3, z3 = adams(x,h)
    plt.figure()
    plt.plot(xtv, tv,color = 'red')
    for i in range(len(x)-1):
        plt.plot([x[i],x[i+1]],[y1[i],y1[i+1]],color = 'blue')
        plt.plot([x[i],x[i+1]],[y3[i],y3[i+1]],color = 'orange')
        plt.plot([x[i],x[i+1]],[y2[i],y2[i+1]],color = 'green')
    plt.grid()
    plt.show()
    return

def main():
    print('Лабораторная работа 4.1\nВариант 14')
    print('(x^2 + 1)y"'," = 2xy'\ny(0) = 1","\ny'(0) = 3\n0 <= x <= 1, h = 0.1")
    x = np.linspace(0.,1.,6)
    h = 0.2
    n = len(x)
    tv = true_val(x)
    
    y, z, dy, dz = runge_kuttt(x,h,n)
    print_runge(x,y,z,dy,dz,tv)
    print(runger(x,h,2))
    
    y, z, dy, dz = eiler(x,h)
    print_eiler(x,y,z,dy,dz,tv)
    print(runger(x,h,1))
    
    y, z = adams(x,h)
    print_adams(x,y,z,tv)
    print(runger(x,h,3))

    plots(x,h)

if __name__ == "__main__":
    main()
