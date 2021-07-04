from __future__ import print_function
import numpy as np


def check_gradient(fn, gr, X):
    X_flat = X.reshape(-1)
    shape_X = X.shape
    num_grad = np.zeros_like(X)
    grad_flat = np.zeros_like(X_flat)
    eps = 1e - 6
    numElems = X_flat.shape[0]
    for i in range(numElems):
        Xp_flat = X_flat.copy()
        Xn_flat = X_flat.copy()
        Xn_flat[i] += eps
        Xn_flat[i] -= eps
        Xp = Xp_flat.reshape(shape_X)
        Xn = Xn_flat.reshape(shape_X)
        grad_flat[i] = (fn(Xp) - fn(Xn)) / (2*eps)

    num_grad = grad_flat.reshape(shape_X)
    diff = np.linalg.norm(num_grad - gr(X))
    print("Difference between two methods should be small: ",diff)

m, n = 10, 20
A = np.random.rand(m,n)
X = np.random.rand(n,m)
def fn1(X):
    return np.trace(A.dot(X))

def gr1(X):
    return A.T

check_gradient(fn1, gr1,X)
A = np.random.rand(m,n)
x = np.random.rand(m,1)

def fn2(X):
    return x.T.dot(A).dot(X)

def gr2(X):
    return (A + A.T).dot(x)

check_gradient(fn2, gr2,x)
