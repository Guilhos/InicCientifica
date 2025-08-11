import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.linalg import block_diag
import os
output_dir = "MPC_to_NN/MPCvsNN - Sistema Compressão/output_figures"
os.makedirs(output_dir, exist_ok=True)

p = 12
m = 3
Nx = 2
Nu = 2
nx = Nx*p
nu = Nu*m
scaling = 1

A = np.array([[-8.83e-1, -3.72e-1],[-4.89e-2,8.29e1]])
B = np.array([[1.20, 7.14e-5],[8.73e-1, 2.93e-5]])

A_ = np.block([[A,B],[np.zeros((Nu,Nu)), np.eye(Nu)]])
B_ = np.block([[np.zeros((Nu,Nu))], [np.eye(Nu)]])

Qtil = np.array([6e-3, 6e-4])
Rtil = np.array([4e-5, 2e-12])

Q = np.diag(np.kron(np.ones(p), Qtil))
R = np.diag(np.kron(np.ones(m), Rtil))

C = np.eye(2)
C_ = np.block([C, np.zeros((Nu,Nu))])
Cu = np.block([np.zeros((Nx,Nx)), np.eye(Nu)])

# Psi
psi = np.block([[C_ @ A_]])
for i in range(p-1):
    psi = np.vstack((psi, psi[-2:] @ A_))

# Theta
theta = np.zeros((nx, nu))
for i in range(p):
    for j in range(i+1):
        if (j+1)*Nu <= nu:
            theta[i*Nu:(i+1)*Nu,j*Nu:(j+1)*Nu] = C_ @ np.linalg.matrix_power(A_, i-j) @ B_
        else:
            continue

# PsiU
psiu = np.block([[Cu @ A_]])
for i in range(p-1):
    psiu = np.vstack((psiu, psiu[-2:] @ A_))

# ThetaU
thetaU = np.zeros((nx, nu))
for i in range(p):
    for j in range(i+1):
        if (j+1)*Nu <= nu:
            thetaU[i*Nu:(i+1)*Nu,j*Nu:(j+1)*Nu] = Cu @ np.linalg.matrix_power(A_, i-j) @ B_
        else:
            continue

H = theta.T @ Q @ theta + R
H = (H + H.T) / 2  # Garantir que H é simétrico
F = np.block([[psi.T @ Q @ theta], [Q @ theta]])

G = np.block([[theta], [-theta], [thetaU], [-thetaU], [np.eye(nx, nu)], [-np.eye(nx, nu)]])
Su = np.block([[psi, np.zeros((nx, nx))],
                [-psi, np.zeros((nx, nx))],
                [psiu, np.zeros((nx, nx))],
                [-psiu, np.zeros((nx, nx))],
                [np.zeros((nx, Nx*Nu)), np.zeros((nx, nx))],
                [np.zeros((nx, Nx*Nu)), np.zeros((nx, nx))]])

# Simulação do MPC

K = 130
x0 = np.array([[8.5], [6.9]])
u0 = np.array([[0.5], [38500]])
xk = x0.copy()
uk = u0.copy()
x_k = np.block([[xk], [uk]])
yspk = (np.ones((p, Nx)) * [8.5, 6.9]).reshape(-1, 1)
z_k = np.block([x_k.T, -yspk.T])

yMax = np.tile(np.array([[12.3], [9.3]]), (p,1))
yMin = np.tile(np.array([[5.5], [5.27]]), (p,1))
uMax = np.tile(np.array([[0.65], [5e4]]), (p,1))
uMin = np.tile(np.array([[0.35], [27e3]]), (p,1))
deltaUMax = np.tile(np.array([[0.1], [2500]]), (p,1))
deltaUMin = np.tile(np.array([[-0.1], [-2500]]), (p,1))

w = np.block([[yMax], [-yMin], [uMax], [-uMin], [deltaUMax], [-deltaUMin]])

deltaU_value = np.zeros((nu, K))
deltaU_mpc = np.zeros(K)
z_k_store_mpc = np.zeros((nx, K))

for j in range(K):
    deltaU = cp.Variable((nu, 1))
    cost = cp.quad_form(deltaU, H) + 2 * z_k @ F @ deltaU
    constraints = [G @ deltaU <= Su @ z_k.T + w]
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()

    deltaU_value[:, j] = deltaU.value

    deltaU_mpc[j] = deltaU_value[:1, j]

    uk = uk + deltaU_mpc[j]
    xk = A @ xk + B @ uk
    x_k = np.block([[xk], [uk]])

    yspk = (np.ones((p, Nx)) * [8.5, 6.9]).reshape(-1, 1)  # Depois alterar o setpoint
    z_k = np.block([[x_k.T @ psi.T], [- yspk.T]]).reshape(-1,1)

    z_k_store_mpc[:, j] = z_k


print('tes')
