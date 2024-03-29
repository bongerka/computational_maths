import numpy as np
from typing import List

def cholesky_solve(A, b):
    n = len(A)
    C = np.zeros((n, n))
    x = np.zeros(n)
    
    # Cholesky decomposition
    for i in range(n):
        for j in range(i+1):
            if i == j:
                temp_sum = 0
                for k in range(j):
                    temp_sum += C[i, k] ** 2
                C[i, j] = np.sqrt(A[i, j] - temp_sum)
            else:
                temp_sum = 0
                for k in range(j):
                    temp_sum += C[i, k] * C[j, k]
                C[i, j] = (A[i, j] - temp_sum) / C[j, j]
    
    # Forward substitution: C^T * y = b
    y = np.zeros(n)
    for i in range(n):
        temp_sum = 0
        for j in range(i):
            temp_sum += C[i, j] * y[j]
        y[i] = (b[i] - temp_sum) / C[i, i]
    
    # Back substitution: C * x = y
    for i in range(n-1, -1, -1):
        temp_sum = 0
        for j in range(i+1, n):
            temp_sum += C[j, i] * x[j]
        x[i] = (y[i] - temp_sum) / C[i, i]
    
    return C, x

def vector_norm(v1, v2):
    return np.linalg.norm(v1 - v2)

def test():
    A1 = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])
    b1 = np.array([1, 2, 3])
    
    A2 = np.array([[1, 2, 3], [2, 8, 11], [3, 11, 42]])
    b2 = np.array([6, 25, 94])
    
    A3 = np.array([[4, 1, 0], [1, 9, -2], [0, -2, 16]])
    b3 = np.array([1, 2, 3])
    
    A4 = np.array([[1, 0, 0], [0, 4, 0], [0, 0, 9]])
    b4 = np.array([1, 2, 3])
    
    A5 = np.array([[25, 15, -5], [15, 18, 0], [-5, 0, 11]])
    b5 = np.array([2, 5, 3])
    
    tests = [(A1, b1), (A2, b2), (A3, b3), (A4, b4), (A5, b5)]
    
    for i, (A, b) in enumerate(tests, start=1):
        x_numpy = np.linalg.solve(A, b)
        
        _, x_custom = cholesky_solve(A, b)
        
        diff_norm = vector_norm(x_numpy, x_custom)
        
        print(f"Test {i}: ", diff_norm)

test()
