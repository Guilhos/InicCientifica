import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import pandas as pd
import casadi as ca
import control as ctrl
from scipy.linalg import block_diag
import os
output_dir = "MPC_to_NN/MPCvsNN - Sistema Compressão/output_figures"
os.makedirs(output_dir, exist_ok=True)

# Sistema

sheet_url = "https://docs.google.com/spreadsheets/d/1r6tQcEkqFMsYhwhe4K3eQ-eAxf3cMnnECnsVwvJnxus/edit?usp=sharing"
url_1 = sheet_url.replace('/edit?usp=sharing', '/export?format=csv')

class Interpolation:
    def __init__(self, file_path, decimal=','):
        self.file_path = file_path
        self.decimal = decimal
        self.params = [-25.0181, 42.0452, -17.9068, 3.0313]

    def load_data(self):
        self.data = pd.read_csv(self.file_path, decimal = self.decimal)
        self.N_rot = np.arange(2e4, 6e4, 1e3)
        self.Mass = np.arange(3, 21.1, 0.1)

        self.Phi = self.data.values   

    def interpolate(self):
        phi_flat = self.Phi.ravel(order='F')
        lut = ca.interpolant('name', 'bspline', [self.N_rot, self.Mass], phi_flat)
        return lut

interp = Interpolation(url_1)
interp.load_data()
lut = interp.interpolate()

A1 = (2.6)*(10**-3)
Lc = 2
kv = 0.38
P1 = 4.5
P_out = 5
C = 479

m_exp = 8.5
p_exp = 6.9
a_exp = 0.5
n_exp = 38500.0

a1 = (P1*(A1/Lc) * (float(lut([n_exp + 1e3, m_exp])) - float(lut([n_exp - 1e3, m_exp])))/(2 * 1e3)) *1e3
a2 = (P1*(A1/Lc) * (float(lut([n_exp, m_exp + 0.1])) - float(lut([n_exp, m_exp - 0.1])))/(2 * 0.1)) *1e3
a3 = -(A1/Lc) * 1e3
a4 = (C**2)/(2)
a5 = -((0.5*kv*500)/(2 * np.sqrt(p_exp * 1000 - P_out * 1000))) * (C**2)/2
a6 = (-kv * np.sqrt(p_exp * 1000 - P_out * 1000)) * (C**2)/2

Ac = np.array([[a2, a3],[a4, a5]])
Bc = np.array([[a1, 0 ],[0, a6]])
Cc = np.eye(2)
Dc = np.zeros((2,2))

Ts = 0.5

sys_c = ctrl.ss(Ac, Bc, Cc, Dc)
sys_d = ctrl.c2d(sys_c, Ts, method='zoh')

# Configurações do MPC

p = 12
m = 3
Nx = 2
Nu = 2
scaling = 1

A, B, C, D = ctrl.ssdata(sys_d)

A_ = np.block([[A,B],[np.zeros((Nu,Nu)), np.eye(Nu)]])
B_ = np.block([[np.zeros((Nu,Nu))], [np.eye(Nu)]])

Qtil = np.array([6e-3, 6e-4])
Rtil = np.array([4e-5, 2e-12])

Q = np.diag(np.kron(np.ones(p), Qtil))
R = np.diag(np.kron(np.ones(m), Rtil))

C_ = np.block([Cc, np.zeros((Nu,Nu))])
Cu = np.block([np.zeros((Nx,Nx)), np.eye(Nu)])

# Psi
psi = np.block([[C_ @ A_]])
for i in range(p-1):
    psi = np.vstack((psi, psi[-2:] @ A_))

# Theta
theta = np.zeros((Nx*p, Nu*m))
for i in range(p):
    for j in range(i+1):
        if (j+1)*Nu <= Nu*m:
            theta[i*Nu:(i+1)*Nu,j*Nu:(j+1)*Nu] = C_ @ np.linalg.matrix_power(A_, i-j) @ B_
        else:
            continue

# PsiU
psiu = np.block([[Cu @ A_]])
for i in range(p-1):
    psiu = np.vstack((psiu, psiu[-2:] @ A_))

# ThetaU
thetaU = np.zeros((Nx*p, Nu*m))
for i in range(p):
    for j in range(i+1):
        if (j+1)*Nu <= Nu*m:
            thetaU[i*Nu:(i+1)*Nu,j*Nu:(j+1)*Nu] = Cu @ np.linalg.matrix_power(A_, i-j) @ B_
        else:
            continue

H = theta.T @ Q @ theta + R
H = (H + H.T) / 2  # Garantir que H é simétrico
F = np.block([[psi.T @ Q @ theta], [Q @ theta]])

G = np.block([[theta], [-theta], [thetaU], [-thetaU], [np.eye(Nx*p, Nu*m)], [-np.eye(Nx*p, Nu*m)]])
Su = np.block([[psi, np.zeros((Nx*p, Nx*p))],
                [-psi, np.zeros((Nx*p, Nx*p))],
                [psiu, np.zeros((Nx*p, Nx*p))],
                [-psiu, np.zeros((Nx*p, Nx*p))],
                [np.zeros((Nx*p, Nx*Nu)), np.zeros((Nx*p, Nx*p))],
                [np.zeros((Nx*p, Nx*Nu)), np.zeros((Nx*p, Nx*p))]])

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

deltaU_value = np.zeros((Nu*p, K))
deltaU_mpc = np.zeros((2, K))
z_k_store_mpc = np.zeros((Nx*p + Nx+Nu, K))

for j in range(K):
    deltaU = cp.Variable((Nu*m, 1))
    cost = cp.quad_form(deltaU, H) + 2 * z_k @ F @ deltaU
    constraints = [G @ deltaU <= Su @ z_k.T + w]
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver="OSQP", verbose=True)

    deltaU_value[:, j] = deltaU.value

    deltaU_mpc[:, j] = deltaU_value[:2, j]

    uk = uk + np.eye(2) @ deltaU_mpc[:, j].reshape(-1,1)
    xk = A @ xk + B @ uk
    x_k = np.block([[xk], [uk]])

    yspk = (np.ones((p, Nx)) * [8.5, 6.9]).reshape(-1, 1)  # Depois alterar o setpoint
    z_k = np.block([x_k.T, -yspk.T])

    z_k_store_mpc[:, j] = z_k


print('tes')
