import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time

@jit(nopython=True)
def seidel(A, b, max_iter=1000, tol=1e-6):
    n = len(b)
    x = np.zeros_like(b)
    
    L = np.tril(A)
    U = np.triu(A, k=1)
    D = np.diag(np.diag(A))
    
    residuals = []
    
    for iter_count in range(max_iter):
        x_old = x.copy()
        
        # Умножение A на x в цикле
        Ax = np.zeros_like(b)
        for i in range(n):
            for j in range(n):
                Ax[i] += A[i, j] * x[j]
        
        # Вычисление нового приближения x
        for i in range(n):
            summation = 0.0
            for j in range(n):
                if i != j:
                    summation += A[i, j] * x[j]
            x[i] = (b[i] - summation) / A[i, i]
        
        residual = np.linalg.norm(Ax - b)
        residuals.append(residual)
        if residual < tol:
            break
            
    return x, np.array(residuals)

def generate_matrix(n):
    A = np.random.rand(n, n)
    A = np.dot(A, A.T) 
    return A

def main():
    n = 150
    A = generate_matrix(n)
    b = np.random.rand(n)
    
    start_time_exact = time.time()
    exact_solution = np.linalg.solve(A, b)
    end_time_exact = time.time()
    exact_error = np.linalg.norm(np.dot(A, exact_solution) - b)
    
    start_time_seidel = time.time()
    seidel_solution, residuals = seidel(A, b)
    end_time_seidel = time.time()
    
    plt.plot(np.log(residuals))
    plt.title('Logarithm of Residuals vs. Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Log Residual')
    plt.grid(True)
    plt.show()
    
    print("Number of iterations:", len(residuals))
    print("Exact error:", exact_error)
    
    exact_time = end_time_exact - start_time_exact
    seidel_time = end_time_seidel - start_time_seidel
    print("Exact solution computation time:", exact_time)
    print("Seidel method computation time:", seidel_time)

if __name__ == "__main__":
    main()
