import numpy as np
import math

def f(x):
    return 1/(16 + x**4)

def trap(f, a, b, h):
	x = np.arange(a,b,h)
	n = len(x)
	res = 0.5*(f(a) + f(b))
	for i in range(1, n):
		res += f(a + i*h)
	res *= h
	return res

def rect(f, a, b, h):
	x = np.arange(a,b,h)
	n = len(x)
	res = f(a+0.5*h)
	for i in range(1, n):
		res += f(a + 0.5*h + i*h)
	res *= h
	return res

def simp(f,a,b,h):
        x = np.arange(a,b,h)
        n = len(x)
        sum2 = 0
        sum4 = 0
        res = 0
        for i in range(1,n,2):
                sum4 += f(a + h*i)
                sum2 += f(a + h*(i+1))
        res = f(a) + 4*sum4 + 2*sum2 - f(b)
        res *= (h/3)
        return res

def rung_f(f,a,b,h1,h2,meth):
        if meth == 1:
                F1 = rect(f,a,b,h1)
                F2 = rect(f,a,b,h2)
                F = F2 + (F2 - F1)/((h1/h2)**1 - 1)
        if meth == 2:
                F1 = trap(f,a,b,h1)
                F2 = trap(f,a,b,h2)
                F = F2 + (F2 - F1)/((h1/h2)**2 - 1)

        if meth == 3:
                F1 = simp(f,a,b,h1)
                F2 = simp(f,a,b,h2)
                F = F2 + (F2 - F1)/((h1/h2)**4 - 1)
        return F

def rung_eps(f,a,b,h1,h2,meth):
        if meth == 1:
                F1 = rect(f,a,b,h1)
                F2 = rect(f,a,b,h2)
                eps = (F2 - F1)/((h1/h2)**1 - 1)
        if meth == 2:
                F1 = trap(f,a,b,h1)
                F2 = trap(f,a,b,h2)
                eps = (F2 - F1)/((h1/h2)**2 - 1)
        if meth == 3:
                F1 = simp(f,a,b,h1)
                F2 = simp(f,a,b,h2)
                eps = (F2 - F1)/((h1/h2)**4 - 1)
        return math.fabs(eps)       
    
def main():
        print('Лабораторная работа 3.5\nВариант 14\ny = 1/(16 + x^4)\na = 0\nb = 2\nh1 = 0.5\nh2 = 0.25')
        print('-----------------------------------------------------------------------------')
        print('|метод           | h1 = 0.5  | h2 = 0.25 |Рунге-Ромберг F|Рунге-Ронберг eps |')
        print('-----------------------------------------------------------------------------')
        print('|прямоугольников |',"%.7f" % rect(f,0,2,0.5),'|',"%.7f" % rect(f,0,2,0.25),'|',"%.7f" % rung_f(f,0,2,0.5,0.25,1),'    |',"%.7f" % rung_eps(f,0,2,0.5,0.25,1),'       |')
        print('-----------------------------------------------------------------------------')
        print('|трапеций        |',"%.7f" % trap(f,0,2,0.5),'|',"%.7f" % trap(f,0,2,0.25),'|',"%.7f" % rung_f(f,0,2,0.5,0.25,2),'    |',"%.7f" % rung_eps(f,0,2,0.5,0.25,2),'       |')
        print('-----------------------------------------------------------------------------')
        print('|Симпсона        |',"%.7f" % simp(f,0,2,0.5),'|',"%.7f" % simp(f,0,2,0.25),'|',"%.7f" % rung_f(f,0,2,0.5,0.25,3),'    |',"%.7f" % rung_eps(f,0,2,0.5,0.25,3),'       |')
        print('-----------------------------------------------------------------------------')


if __name__ == "__main__":
    main()
