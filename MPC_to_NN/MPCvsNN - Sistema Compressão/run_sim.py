import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.linalg import block_diag
import os
output_dir = "MPC_to_NN/MPCvsNN - Sistema Compressão/output_figures"
os.makedirs(output_dir, exist_ok=True)

p = 12
m = 3
N = 2
nx = 2*p
nu = 2*m
scaling = 1

A = np.array([[-8.83e-1, -3.72e-1],[-4.89e-2,8.29e1]])
B = np.array([[1.20, 7.14e-5],[8.73e-1, 2.93e-5]])

A_ = np.block([[A,B],[np.zeros((N,N)), np.eye(N)]])
B_ = np.block([[np.zeros((N,N))], [np.eye(N)]])

Qtil = np.array([6e-3, 6e-4])
Rtil = np.array([4e-5, 2e-12])

Q = np.diag(np.kron(np.ones(p), Qtil))
R = np.diag(np.kron(np.ones(m), Rtil))

C = np.eye(2)
C_ = np.block([C, np.zeros((N,N))])
Cu = np.block([np.zeros((N,N)), np.eye(N)])

# Psi
psi = np.block([[C_ @ A_]])
for i in range(p-1):
    psi = np.vstack((psi, psi[-2:] @ A_))

# Theta
theta = np.zeros((nx, nu))
for i in range(p):
    for j in range(i+1):
        if (j+1)*N <= nu:
            theta[i*N:(i+1)*N,j*N:(j+1)*N] = C_ @ np.linalg.matrix_power(A_, i-j) @ B_
        else:
            continue

H = theta.T @ Q @ theta + R
F = np.block([[psi.T @ Q @ theta], [Q @ theta]])

# Simulação do MPC

K = 130
#x0 = 

print('tes')
