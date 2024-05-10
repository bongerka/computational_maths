import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def f(x, a):
    q = np.array([1, np.log(x), x**-2, x**-1, x, x**2, x**3], dtype="object")
    return np.dot(a, q)

def v(x, b):
    v = np.array([1, x, np.sqrt(x), x**2, x**3, x**4, x**6], dtype="object")
    return np.dot(b, v)

def scalar_product(i, j, xl, xr):
    def integrand(x):
        return v(x, np.eye(7)[i]) * v(x, np.eye(7)[j])
    return quad(integrand, xl, xr)[0]

def find_coefficients(a, xl, xr):
    n = 7
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            G[i, j] = scalar_product(i, j, xl, xr)
    
    b = np.zeros(n)
    for i in range(n):
        def integrand(x):
            return f(x, a) * v(x, np.eye(7)[i])
        b[i] = quad(integrand, xl, xr)[0]
    
    return np.linalg.solve(G, b)

def plot_results(a, xl, xr):
    x = np.linspace(xl, xr, 100)
    b = find_coefficients(a, xl, xr)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    ax1.plot(x, f(x, a), label='Original')
    ax1.plot(x, v(x, b), label='Approximation')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f'Approximation on [{xl}, {xr}]')
    ax1.legend()

    ax2.plot(x, f(x, a) - v(x, b), label='Difference')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title(f'Difference on [{xl}, {xr}]')
    ax2.legend()

    plt.tight_layout()
    plt.show()

a = np.array([1, -2, 3, -4, 5, -6, 7])

xl, xr = 0.1, 1
plot_results(a, xl, xr)

xl, xr = 1, 5
plot_results(a, xl, xr)

