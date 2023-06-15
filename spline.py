import numpy as np
import math as math
import matplotlib.pyplot as plt
from qr import *

y = np.array([1, 4, 7, 4, 12])
x = np.array([1, 2, 3, 4, 5])





def Filling_of_diagonals(n, x, y):
    low_d = [0] * (n - 2)
    main_d = [0] * (n - 2)
    upper_d = [0] * (n - 2)
    free_ch = [0] * (n - 2)
    h = [0] * n
    for i in range(1, n):
        h[i] = x[i] - x[i - 1]
    for i in range(n - 2):
        if i != 0 and i != n - 3:
            low_d[i] = h[i + 1] / 6
            main_d[i] = (h[i + 1] + h[i + 2]) / 3
            upper_d[i] = h[i + 2] / 6
        elif i == 0:
            main_d[i] = (h[i + 1] + h[i + 2]) / 3
            upper_d[i] = h[i + 2] / 6
        else:
            low_d[i] = h[i + 1] / 6
            main_d[i] = (h[i + 1] + h[i + 2]) / 3

        free_ch[i] = ((y[i + 2] - y[i + 1]) / h[i + 2]) - ((y[i + 1] - y[i]) / h[i + 1])

    return low_d, main_d, upper_d, free_ch


def Finding_gamma(n, low_d, main_d, upper_d, free):
    for i in range(1, n - 1):
        m = low_d[i] / main_d[i - 1]
        main_d[i] = main_d[i] - m * upper_d[i - 1]
        free[i] = free[i] - m * free[i - 1]

    gamma = [0] * n
    gamma[n - 1] = free[n - 2] / main_d[n - 2]

    for i in range(n - 2, -1, -1):
        gamma[i] = (free[i] - upper_d[i] * gamma[i + 1]) / main_d[i]

    return gamma


def Spline(n, gamma, x, y, f_x):
    h = [0] * n
    for i in range(1, n):
        h[i] = x[i] - x[i - 1]

    for i in range(n):
        if f_x >= x[i] and f_x <= x[i + 1]:
            return y[i] * ((x[i + 1] - f_x) / h[i + 1]) + y[i + 1] * ((f_x - x[i]) / h[i + 1]) + gamma[i] * ((pow((x[i + 1] - f_x), 3) - pow(h[i + 1], 2) * (x[i + 1] - f_x)) / (6 * h[i + 1])) + gamma[i + 1] * ((pow((f_x - x[i]), 3) - pow(h[i + 1], 2) * (f_x - x[i])) / (6 * h[i + 1]))

def fill_koaf(x, alfa, n, g):
    h = [0] * n
    a = [0] * (n - 1)
    b = [0] * (n - 1)
    c = [0] * (n - 1)
    d = [0] * (n - 1)
    for i in range(1, n):
        h[i] = x[i] - x[i - 1]
    for i in range(n-1):
        a[i] = (alfa[i+1] - alfa[i])/(6*h[i+1])
        b[i] = (x[i+1]*alfa[i] - x[i]*alfa[i+1])/(2*h[i+1])
        c[i] = ((g[i+1] - g[i])/h[i+1]) + (((x[i])**2*alfa[i+1] - (x[i+1])**2*alfa[i])/(2*h[i+1])) + ((h[i+1]*alfa[i] - h[i+1]*alfa[i+1])/6)
        d[i] = ((x[i+1]*g[i] - x[i]*g[i+1])/h[i+1]) + (((x[i+1])**3*alfa[i] - (x[i])**3*alfa[i+1])/(6*h[i+1])) + ((h[i+1]*x[i]*alfa[i+1] - h[i+1]*x[i+1]*alfa[i])/6)
    return a, b, c, d
def solve_eq(a, b, c, d, n, x, g, y,  alfa):
    for i in range(n-1):
        if a[i] != 0:
            A = np.array([[0, 1, 0], [0, 0, 1], [-d[i]/ a[i], -c[i]/a[i], -b[i]/a[i]]])
            eq = qr_householder(A)
            for i in eq:
                if i <= x[n-1] and i >= x[0]:
                    if abs(Spline(n, alfa, x, g, i)) < 0.0001:
                        print("Coordinate of crossing: y = ", Spline(n, alfa, x, y, i), " x = ", i)


            #print(eq)


y = np.array([0.0, 0.125, 1.0, 3.375, 10.0, 5.625])
x = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
n1 = 6
low_d, main_d, upper_d, free_ch = Filling_of_diagonals(n1,  x, y)
gamma = Finding_gamma(n1-1, low_d, main_d, upper_d, free_ch)
gamma_f1 = np.array([gamma[i] for i in range(1, n1-2)])

y_2 = np.array([0.3, 0.12, 2.8, 9.35, 3.0, 0.0])

n2 = 6
low_d_2, main_d_2, upper_d_2, free_ch_2 = Filling_of_diagonals(n2,  x, y_2)
gamma_2 = Finding_gamma(n2-1, low_d_2, main_d_2, upper_d_2, free_ch_2)
g = np.array([0]*n1)
alfa = np.array([0]*n1)

for i in range(n1):
    g[i] = y[i] - y_2[i]

for i in range(n1-2):
    alfa[i] = gamma[i] - gamma_2[i]

a, b, c, d = fill_koaf(x, alfa, n1, g)
solve_eq(a, b, c, d, n1, x, g, y, alfa)
'''xl = np.linspace(0, 2.0, 1000)

yl = [Spline(n1, alfa, x, g, i) for i in xl]
plt.plot(xl, yl)
plt.show()'''

