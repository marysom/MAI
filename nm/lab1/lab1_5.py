from numpy import *
from numpy.linalg import qr
import numpy as np
from math import sqrt,fabs

def qr_dec(A,count):
    m, n = A.shape
    Q = np.eye(m)
    for i in range(n - (m == n)):
        H = np.eye(m)
        H[i:, i:] = make_householder(A[i:, i])
        Q = np.dot(Q, H)
        A = np.dot(H, A)
    return Q, A
 
def make_householder(a):
    v = a / (a[0] + np.copysign(np.linalg.norm(a), a[0]))
    v[0] = 1
    H = np.eye(a.shape[0])
    H -= (2 / np.dot(v, v)) * np.dot(v[:, None], v[None, :])
    return H

def qr_alg(A,n,eps):
    k = 0
    R = np.array(A)
    it = 0   
    while True:
        it += 1
        e_k = 0
        Q, A = qr_dec(A,n)
        A = np.dot(A, Q)
        print(it,'итерация\nA = ',A)
        e_k = A[2,0]
        '''
        for m in range(1, n):
            e_k += A[m][0] ** 2            
        e_k = sqrt(e_k)'''
        print('epsk = ',e_k,'\n')
        if e_k < eps:
           break           
    D = ((A[1][1] + A[2][2]) ** 2 - 4 * (A[1][1] * A[2][2] - A[1][2] * A[2][1]))
    l = []
    if D >= 0:
        for i in range(n):
            l.append("x{} {}\n".format(i + 1, A[i][i]))
    else:
        l.append("x1 = {}".format(A[0][0]))
        l.append("x2 = {}".format(complex((A[1][1] + A[2][2])/2, sqrt(fabs(D))/2)))   
        l.append("x3 = {}".format(complex((A[1][1] + A[2][2])/2, -sqrt(fabs(D))/2)))  
    for i in range(len(l)):
            print(l[i] )
    return

def norm(vec, i, n):
    res = 0
    for j in range(i, n):
        res += vec[j] ** 2
    return sqrt(res)

def mult(a,n):
    first = np.zeros((n, n))
    first[0] = a
    return np.dot(np.transpose(first), first)

def inputdata():
    f = open('input_lab1_5.txt')
    line = f.readlines()
    n=len(line)
    a=line[0].split()
    a=list(map(float,a))
    for i in range(1,n):
        tmp=line[i].split()
        tmp=list(map(float,tmp))
        a=np.row_stack((a,tmp))
    f.close()
    return a, n


def main():
    print('Лабораторная работа 1.5')
    print('QR-алгоритм')
    print('Вариант 14')
    a, n = inputdata()
    print('А=',a)
    print('Введите точность вычислений:')
    eps=float(input())
    print('-----------------------------------------------')
    #a = [[1.,3.,1.],[1.,1.,4],[4.,3.,1]]
    (q1,r1) = linalg.qr(a)
    print('QR-разложение функцией из numpy: \nQnp = ',q1,'\nRnp = ',r1)
    (q,r) = qr_dec(a,n)
    print('\nQR-разложение: \nQ = ',q,'\nR = ',r)
    print('-----------------------------------------------')
    qr_alg(a,n,eps)

if __name__ == "__main__":
    main()
