import numpy as np
from itertools import product, tee
import random
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import math
import time

def rec(f, a, b, n):
    h = (b - a)/n
    result = f(a+0.5*h)
    for i in range(1, n):
        result += f(a + 0.5*h + i*h)
    result *= h
    return result

def rec2_1(f, a, b, c, d, nx, ny):
    t = time.clock()
    hx = (b - a)/nx
    hy = (d - c)/ny
    I = 0
    for i in range(nx):
        for j in range(ny):
            xi = a + hx/2 + i*hx
            yj = c + hy/2 + j*hy
            I += hx*hy*f(xi,yj)
    t = time.clock() - t
    return I, t

def rec2_2(f, a, b, c, d, nx, ny):
    t = time.clock() 
    g = lambda x: rec(lambda y: f(x, y), c, d, ny)
    result = rec(g, a, b, nx)
    t = time.clock() - t
    return result, t

def rec3_1(g, a, b, c, d, e, f, nx, ny, nz):
    t = time.clock()
    hx = (b - a)/nx
    hy = (d - c)/ny
    hz = (f - e)/nz
    I = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                xi = a + hx/2 + i*hx
                yj = c + hy/2 + j*hy
                zk = e + hz/2 + k*hz
                I += hx*hy*hz*g(xi,yj,zk)
    t = time.clock() - t
    return I, t

def rec3_2(g, a, b, c, d, e, f, nx, ny, nz):
    t = time.clock()
    p = lambda x, y: rec(lambda z: g(x, y, z), e, f, nz)
    q = lambda x: rec(lambda y: p(x, y), c, d, ny)
    result = rec(q, a, b, nx)
    t = time.clock() - t
    return result, t

def rec5_2(g, aa,ab, ac,ad, ae,af, ag,ah, ai,aj, nx, ny, nz, np, nq):
    t = time.clock()
    pp = lambda x, y, z, p: rec(lambda q: g(x, y, z, p, q), ai, aj, nq)
    qq = lambda x, y, z: rec(lambda p: pp(x, y, z, p), ag, ah, np)
    cc = lambda x, y: rec(lambda z: qq(x, y, z), ae, af, nz)
    bb = lambda x: rec(lambda y: cc(x,y), ac, ad, ny)
    result = rec(bb, aa, ab, nx)
    t = time.clock() - t
    return result, t

def rec5_1(g, aa,ab, ac,ad, ae,af, ag,ah, ai,aj, nx, ny, nz, np, nq):
    t = time.clock()
    hx = (ab - aa)/nx
    hy = (ad - ac)/ny
    hz = (af - ae)/nz
    hp = (ah - ag)/np
    hq = (aj - ai)/nq
    I = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for r in range(np):
                    for u in range(nq):
                        xi = aa + hx/2 + i*hx
                        yj = ac + hy/2 + j*hy
                        zk = ae + hz/2 + k*hz
                        pr = ag + hp/2 + r*hp
                        qu = ai + hq/2 + u*hq
                        I += hx*hy*hz*hp*hq * g(xi,yj,zk,pr,qu)
    t = time.clock() - t
    return I, t


def mc(f, dim, n):
    t = time.clock()
    result = 0
    for _ in range(n):
        x = [d[0]+random.random()*(d[1]-d[0]) for d in dim]
        if f(x):
            result += f(x)
    for d in dim:
        result *= d[1] - d[0]
    t = time.clock() - t
    return result / n, t

def mcd(f, cond, dim, n): 
    val = 0
    for _ in range(n):
        x = [d[0]+random.random()*(d[1]-d[0]) for d in dim]
        if cond(x):
            val += cond(x)
    for d in dim:
        val *= d[1] - d[0]
    return val / n

def main():
    print('Курсовая работа\nВариант 9')
    print('Вычисление многократных интегралов с использованием квадратурных формул и метода Монте-Карло')
#====================
    f = lambda x, y: 2*x + y
    t = time.clock()
    rec21, t21 = rec2_1(f, 0, 1, 1, 2, 100, 100)
    t21 = time.clock() - t
    t = time.clock()
    rec22, t22 = rec2_2(f, 0, 1, 1, 2, 100, 100)
    t22 = time.clock() - t
    g = lambda x, y: (1 if (0 <= x <= 1) and (1 <= y <= 2) else -1)
    f = lambda x: 2*x[0] + x[1]
    t = time.clock()
    mc2, t2 = mc(f,[(0,1),(1,2)],10)
    t2 = time.clock() - t

    rungFr = mc2 + (mc2 - rec21)/((1000/5)**1 - 1)
    rungEr = math.fabs((mc2 - rec21)/((1000/5)**1 - 1))

    print('f(x,y) = 2x + y\n 0<=x<=1  1<=y<=2 ')
    table = PrettyTable(['rec1', 'rec2','mc2','rungeF rec','rungeE rec'])
    table.add_row([rec21, rec22,  mc2, "%.6f" %rungFr, "%.4f" %rungEr])
    print(table)

#==========================
    g = lambda x, y, z: 2*x + y - 4*z
    t = time.clock()
    rec31, t31 = rec3_1(g,0,2,2,3,-1,2,50,50,50)
    t31 = time.clock() - t
    t = time.clock()
    rec32, t32 = rec3_2(g,0,2,2,3,-1,2,50,50,50)
    t32 = time.clock() - t

    g = lambda x: 2*x[0] + x[1] - 4*x[2]
    t = time.clock()
    mc3, t3 = mc(g,[(0,2),(2,3),(-1,2)],50)
    t3 = time.clock() - t

    rungFr = mc3 + (mc3 - rec31)/((1000/5)**1 - 1)
    rungEr = math.fabs((mc3 - rec31)/((1000/5)**1 - 1))

    print('f(x,y,z) = 2x + y - 4z\n 0<=x<=2  2<=y<=3  -1<=z<=2')
    table = PrettyTable(['rec1', 'rec2','mc2','rungeF rec','rungeE rec'])
    table.add_row([rec31, rec32, mc3, "%.6f" %rungFr, "%.4f" %rungEr])
    print(table)

#============================
    g = lambda x, y, z, p, q: 2*x +y - 4*z + p + q
    f = lambda x, y, z, p, q: (1 if (0<=x<=2) and (2<=y<=3) and (-1<=z<=2) and (1<=p<=6) and(-8<=q<=7) else -1)
    #print(mc_5(g, f, 0,2, 2,3, -1,2, 1,6, -8,7, 10))
    t = time.clock()
    rec51, t51 = rec5_1(g, 0,2, 2,3, -1,2, 1,6, -8,7, 10,10,10,10,10)
    t51 = time.clock() - t
    t = time.clock()
    rec52, t52 = rec5_2(g, 0,2, 2,3, -1,2, 1,6, -8,7, 10,10,10,10,10)
    t52 = time.clock() - t
    g = lambda x: 2*x[0] + x[1] - 4*x[2] + x[3] + x[4]
    t = time.clock()
    mc5, t5 = mc(g,[(0,2),(2,3),(-1,2),(1,6),(-8,7)],30)
    t5 = time.clock() - t

    rungFr = mc5 + (mc5 - rec51)/((1000/5)**1 - 1)
    rungEr = math.fabs((mc5 - rec51)/((10000/10)**1 - 1)) 
    print('f(x,y,z) = 2x + y - 4z + p + q\n 0<=x<=2  2<=y<=3  -1<=z<=2, 1<=p<=6, -8<=q<=7')
    table = PrettyTable(['rec1', 'rec2','mc5','rungeF rec','rungeE rec'])
    table.add_row([rec51, rec52, mc5, "%.6f" %rungFr, "%.4f" %rungEr])
    print(table)
#====================
    print('Интегрирование по области сложной формы методом Монте-Карло')
    print('f(x,y,z) = 1\n G = x^2 + y^2 + z^2 < 1')
    f = lambda x: 1
    g = lambda x: x[0]**2 + x[1]**2 + x[2]**2 < 1
    m = mcd(f,g,[(-1,1),(-1,1),(-1,1)],30)
    print(m)
    print('f(x,y,z,p,q) = 2*x +y - 4*z + p + q\n G = x^2 + y^2 + z^2 - pq < 1')
    f = lambda x: 2*x[0] + x[1] - 4*x[2] + x[3] + x[4]
    g = lambda x: x[0]**2 + x[1]**2 + x[2]**2 + x[3]*x[4]<1
    m = mcd(f,g,[(-1,1),(-1,1),(-1,1),(-1,1),(-1,1)],30)
    print(m)    

    plt.figure()
    plt.grid()
    x = [2.,3.,5.]
    y1 = [t21, t31, t51]
    y2 = [t22, t32, t52]
    y3 = [t2, t3, t5]

    plt.scatter(x, y1, color = 'red', s = 10)
    plt.plot(x,y1,color = 'red')
    plt.scatter(x, y2, color = 'blue', s = 10)
    plt.plot(x,y2,color = 'blue')
    plt.scatter(x, y3, color = 'green', s = 10)
    plt.plot(x,y3,color = 'green')
    plt.xlabel('кратность интеграла')
    plt.ylabel('время вычисления интеграла')
    plt.show()

if __name__ == '__main__':
    main()

