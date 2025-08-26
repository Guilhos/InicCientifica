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
np.set_printoptions(precision=2)

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
Vp = 2

m_exp = 7.745
p_exp = 6.662
a_exp = 0.5
n_exp = 3.85

# dm/dt = a11 * m + a12 * p + b12 * n
a11 = P1 * A1/Lc * 1e6 * (float(lut([n_exp*1e4 , m_exp + 0.1])) - float(lut([n_exp*1e4 , m_exp - 0.1]))) / 2 / 0.1
a12 = -A1 / Lc * 1e6 
b12 = P1 * A1/Lc * 1e10 * (float(lut([n_exp*1e4 + 1e3, m_exp])) - float(lut([n_exp*1e4  - 1e3, m_exp])))/ 2 / 1e3

# dp/dt = a21 * m + a22 * p + b21 * a
a21 = C**2 / Vp/1e6
a22 = -C**2 / Vp * a_exp * kv *500 / np.sqrt((p_exp - P_out) * 1000)/1e6
b21 = -C**2 / Vp * kv * np.sqrt((p_exp - P_out) * 1000)/1e6

Ac = np.array([[a11, a12],
               [a21, a22]])
Bc = np.array([[0, b12 ],
               [b21, 0]])
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
B_ = np.block([[B], [np.eye(Nu)]])

Qtil = np.array([2, 2])
Rtil = np.array([10, 10])

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
F = np.block([[psi.T @ Q @ theta], [np.kron(np.ones((p,1)), np.eye(Nx)).T @ -Q @ theta]])

G = np.block([[theta],
                [-theta],
                [thetaU],
                [-thetaU],
                [np.eye(Nx*p, Nu*m)],
                [-np.eye(Nx*p, Nu*m)]])
Su = np.block([[-psi, np.zeros((Nx*p, Nx))],
                [psi, np.zeros((Nx*p, Nx))],
                [-psiu, np.zeros((Nx*p, Nx))],
                [psiu, np.zeros((Nx*p, Nx))],
                [np.zeros((Nx*p, Nx*Nu)), np.zeros((Nx*p, Nx))],
                [np.zeros((Nx*p, Nx*Nu)), np.zeros((Nx*p, Nx))]])

# Simulação do MPC

K = 100
x_op = np.array([[m_exp], [p_exp]])    
u_op = np.array([[a_exp], [n_exp]])   
y_op = x_op.copy()

x0 = np.array([[7.745], [6.662]]) - x_op
u0 = np.array([[0.5], [3.85]]) - u_op

xk = x0.copy()
uk = u0.copy()

x_k = np.block([[xk], [uk]])
yspk = np.array([[7.745], [6.662]] - y_op)
z_k = np.block([x_k.T, yspk.T])

yMax = np.tile(np.array([[12.3], [9.3]]) - y_op, (p,1))
yMin = np.tile(np.array([[5.5], [5.27]]) - y_op, (p,1))
uMax = np.tile(np.array([[0.65], [5]]) - u_op, (p,1))
uMin = np.tile(np.array([[0.35], [2.7]]) - u_op, (p,1))
deltaUMax = np.tile(np.array([[0.1], [0.25]]), (p,1))
deltaUMin = np.tile(np.array([[-0.1], [-0.25]]), (p,1))

w = np.block([[yMax],
                [-yMin],
                [uMax],
                [-uMin],
                [deltaUMax],
                [-deltaUMin]])

deltaU_value = np.zeros((K, Nu*m))
deltaU_mpc = np.zeros((K, Nu))
z_k_store_mpc = np.zeros((K, Nx+Nx+Nu))

for j in range(K):
    deltaU = cp.Variable((Nu*m, 1))
    cost = cp.quad_form(deltaU, H) + 2 * z_k @ F @ deltaU
    Su_zk = Su @ z_k.T
    Su_zk_w = Su @ z_k.T + w
    constraints = [G @ deltaU <= Su @ z_k.T + w]
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver="OSQP", verbose=True)

    deltaU_value[j, :] = deltaU.value.flatten()

    deltaU_mpc[j, :] = deltaU_value[j, :2]

    uk = uk + np.eye(2) @ deltaU_mpc[j, :].reshape(-1,1)
    xk = A @ xk + B @ uk 
    x_k = np.block([[xk], [uk]])

    if j == 0:
        yspk = np.array([[7.745], [6.662]]- y_op)
    elif j == 10:
        yspk = np.array([[8.5], [7]]- y_op)
    elif j == 40:
        yspk = np.array([[7], [6]]- y_op)
    elif j == 70:
        yspk = np.array([[7.5], [6.5]]- y_op)

    z_k = np.block([x_k.T, yspk.T])

    z_k_store_mpc[j, :] = z_k

# Simulação da rede neural implicita

iters = 25
y_store = np.zeros((Nx*p*Nu*m, iters))
phi_store = np.zeros((Nx*p*Nu*m, iters))
res_store = np.zeros((Nx*p*Nu*m, iters))
res_norm = np.zeros(iters)
sign_store = np.ones((Nx*p*Nu*m, iters))

S = Su + G @ np.linalg.inv(H) @ F.T
D = np.eye(Nx*p*6) - G @ np.linalg.inv(H) @ G.T
D_norm = np.linalg.norm(D, 2)

def compute_D(QtilA, RtilA, pA, mA, A_, B_, C_, Cu, Su, theta, psi, thetaU, psiu, Nx, Nu):

    psiA = np.block([[C_ @ A_]])
    for i in range(pA-1):
        psiA = np.vstack((psiA, psiA[-2:] @ A_))

    thetaA = np.zeros((Nx*pA, Nu*mA))
    for i in range(pA):
        for j in range(i+1):
            if (j+1)*Nu <= Nu*mA:
                thetaA[i*Nu:(i+1)*Nu,j*Nu:(j+1)*Nu] = C_ @ np.linalg.matrix_power(A_, i-j) @ B_
            else:
                continue
    
    # PsiU
    psiuA = np.block([[Cu @ A_]])
    for i in range(pA-1):
        psiuA = np.vstack((psiu, psiu[-2:] @ A_))

    # ThetaU
    thetaUA = np.zeros((Nx*pA, Nu*mA))
    for i in range(pA):
        for j in range(i+1):
            if (j+1)*Nu <= Nu*mA:
                thetaUA[i*Nu:(i+1)*Nu,j*Nu:(j+1)*Nu] = Cu @ np.linalg.matrix_power(A_, i-j) @ B_
            else:
                continue

    # Q e R ampliados
    QA = np.diag(np.kron(np.ones(pA), QtilA))
    RA = np.diag(np.kron(np.ones(mA), RtilA))

    # Matriz H e F
    HA = thetaA.T @ QA @ thetaA + RA
    HA = (HA + HA.T) / 2  # garantir simetria
    FA = np.block([[psiA.T @ QA @ thetaA],
                  [np.kron(np.ones((pA,1)), np.eye(Nx)).T @ -QA @ thetaA]])

    # Matriz G
    GA = np.block([[thetaA],
                  [-thetaA],
                  [thetaUA],
                  [-thetaUA],
                  [np.eye(Nx*pA, Nu*mA)],
                  [-np.eye(Nx*pA, Nu*mA)]])

    # Matriz D
    DA = np.eye(Nx*pA*6) - GA @ np.linalg.inv(HA) @ GA.T
    D_normA = np.linalg.norm(DA, 2)
    eigvalsA = np.linalg.eigvals(DA)
    return D_normA, eigvalsA

# Teste de diferentes sintonias
Qtil_testes = [[1, 1], [1e3, 1e3], [1e6, 1e6], [1e9, 1e9]]
Rtil_testes = [[1, 1], [1e3, 1e3], [1e6, 1e6], [1e9, 1e9]]
p_testes = [10, 15, 20]
m_testes = [3, 5, 10]

for QtilA in Qtil_testes:
    for RtilA in Rtil_testes:
        for pA in p_testes:
            for mA in m_testes:
                D_normA, eigvals = compute_D(QtilA, RtilA, pA, mA, A_, B_, C_, Cu, Su, theta, psi, thetaU, psiu, Nx, Nu)
                print(f"{QtilA};{RtilA};{pA};{mA};{D_normA}\";\"{max(np.real(eigvals))}")

xk = x0.copy()
uk = u0.copy()
x_k = np.block([[xk], [uk]])
yspk = np.array([[7.745], [6.662]] - y_op)
z_k = - np.block([x_k.T, yspk.T])

c_MPC = S @ z_k.T + w
zeta = -c_MPC

y0 = np.zeros((Nx*p*Nu*m, 1))
phi = np.zeros((Nx*p*Nu*m, 1))
for g in range(Nx*p):
    if y0[g] >= 0:
        phi[g] = y0[g]
    else:
        phi[g] = 0

residual = y0 - D @ phi - zeta
K_gain = 1.1 * np.eye(Nx*p*Nu*m)
y = y0.copy()

deltaU_nn = np.zeros((K, Nu))
deltaU_ramp = np.zeros((K, Nu*m))
z_k_store_nn = np.zeros((K, Nx + Nx + Nu))
res_norm_store = np.zeros((K, iters))

for k in range(K):
    for i in range(iters):
        c_MPC = S @ z_k.T + w
        zeta = -c_MPC
        y_store[:, i:i+Nx] = y
        phi_store[:, i:i+Nx] = phi
        res_store[:, i:i+Nx] = residual
        ykp1 = D @ phi + zeta + K_gain @ residual

        for g in range(Nx*p*Nu*m):
            if ykp1[g] > 0:
                phi[g] = ykp1[g]
            else:
                phi[g] = 0
                sign_store[g, i] = -1
        
        y = ykp1.copy()
        residual = y - D @ phi - zeta
        res_norm[i] = np.linalg.norm(residual)
        res_norm_store[j, i] = np.linalg.norm(y - D @ phi - zeta)

    y_ramp = ykp1.copy()
    deltaU_ramp[k, :] = (-np.linalg.inv(H) @ (F.T @ z_k.T + G.T @ phi)).flatten()

    deltaU_nn[k, :] = deltaU_ramp[k, :2]
    uk = uk + np.eye(2) @ deltaU_nn[k, :].reshape(-1,1)
    #print(k)
    xk = A @ xk + B @ uk
    x_k = np.block([[xk], [uk]])

    if j == 0:
        yspk = np.array([[7.745], [6.662]] - y_op)
    elif j == 10:
        yspk = np.array([[8.5], [7]] - y_op)
    elif j == 40:
        yspk = np.array([[7], [6]] - y_op)
    elif j == 70:
        yspk = np.array([[7.5], [6.5]] - y_op)

    z_k = np.block([x_k.T, yspk.T])
    z_k_store_nn[k, :] = z_k

# Plots
t = np.arange(K) * Ts
nT = len(t)
plt.figure(figsize=(10, 6))
plt.plot(t,z_k_store_mpc[:nT, 0] + x_op[0], label='Massa (kg)')
plt.plot(t,z_k_store_mpc[:nT, 4] + x_op[0], label='Setpoint', linestyle='-.', color = 'red')
plt.plot(t,z_k_store_nn[:nT, 0] + x_op[0], label='Massa NN (kg)', linestyle='--')
plt.plot(t,np.ones((nT,1)) * (yMax[0] + x_op[0]), label='Massa Max', linestyle=':', color='black')
plt.plot(t,np.ones((nT,1)) * (yMin[0] + x_op[0]), label='Massa Min', linestyle=':', color='black')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'massa_mpc.png'))

plt.figure(figsize=(10, 6))
plt.plot(t,z_k_store_mpc[:nT, 1] + x_op[1], label='Pressão (MPa)')
plt.plot(t,z_k_store_mpc[:nT, 5] + x_op[1], label='Setpoint', linestyle='-.', color = 'red')
plt.plot(t,z_k_store_nn[:nT, 1] + x_op[1], label='Pressão NN (MPa)', linestyle='--')
plt.plot(t,np.ones((nT,1)) * (yMax[1] + x_op[1]), label='Pressão Max', linestyle=':', color='black')
plt.plot(t,np.ones((nT,1)) * (yMin[1] + x_op[1]), label='Pressão Min', linestyle=':', color='black')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'pressao_mpc.png'))

plt.figure(figsize=(10, 6))
plt.plot(t, z_k_store_mpc[:nT, 2] + u_op[0], label='Abertura da válvula (m)')
plt.plot(t, z_k_store_nn[:nT, 2] + u_op[0], label='Abertura da válvula NN (m)', linestyle='--')
plt.plot(t, np.ones((nT,1)) * (uMax[0] + u_op[0]), label='Abertura Max', linestyle=':', color='black')
plt.plot(t, np.ones((nT,1)) * (uMin[0] + u_op[0]), label='Abertura Min', linestyle=':', color='black')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'abertura_valvula_mpc.png'))

plt.figure(figsize=(10, 6))
plt.plot(t, z_k_store_mpc[:nT, 3] + u_op[1], label='Rotação do motor (rpm)')
plt.plot(t, z_k_store_nn[:nT, 3] + u_op[1], label='Rotação do motor NN (rpm)', linestyle='--')
plt.plot(t, np.ones((nT,1)) * (uMax[1] + u_op[1]), label='Rotação Max', linestyle=':', color='black')
plt.plot(t, np.ones((nT,1)) * (uMin[1] + u_op[1]), label='Rotação Min', linestyle=':', color='black')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'rotacao_motor_mpc.png'))

D_eigvals = np.linalg.eigvals(D)
print(D_norm)
