import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import pandas as pd
import casadi as ca
import control as ctrl
from scipy.linalg import block_diag
import os
output_dir = "MPC_to_NN/MPCvsNN - Sistema Compressão/output_figures/pos_drummond"
os.makedirs(output_dir, exist_ok=True)
np.set_printoptions(precision=5)

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
C = 479
Vp = 2
P1 = 4.5
P_out = 5

m_max, m_min = 12.3, 5.5
p_max, p_min = 9.3, 5.27
a_max, a_min = 0.65, 0.35
n_max, n_min = 5, 2.7

def normalize(value, var):
    if var == 'm':
        min_val, max_val = m_min, m_max
    elif var == 'p':
        min_val, max_val = p_min, p_max
    elif var == 'a':
        min_val, max_val = a_min, a_max
    elif var == 'n':
        min_val, max_val = n_min, n_max

    return (value - min_val) / (max_val - min_val)

def denormalize(norm_value, var):
    if var == 'm':
        min_val, max_val = m_min, m_max
    elif var == 'p':
        min_val, max_val = p_min, p_max
    elif var == 'a':
        min_val, max_val = a_min, a_max
    elif var == 'n':
        min_val, max_val = n_min, n_max

    return norm_value * (max_val - min_val) + min_val

m_exp = normalize(7.745, 'm')
p_exp = normalize(6.662, 'p')
a_exp = normalize(0.5, 'a')
n_exp = normalize(3.85, 'n')

# dm/dt = a11 * m + a12 * p + b12 * n
a11 = P1 * A1/Lc * 1e6 * (float(lut([denormalize(n_exp,'n')*1e4, denormalize(m_exp,'m') + 0.1])) - float(lut([denormalize(n_exp,'n')*1e4, denormalize(m_exp,'m') - 0.1]))) / 2 / 0.1
a12 = -A1 * (p_max - p_min) * 1e6 / Lc / (m_max - m_min)
b12 = P1 * A1/Lc * 1e10 * (n_max - n_min) / (m_max - m_min) * (float(lut([denormalize(n_exp,'n')*1e4 + 1e3, denormalize(m_exp,'m')])) - float(lut([denormalize(n_exp,'n')*1e4  - 1e3, denormalize(m_exp,'m')]))) / 2 / 1e3

# dp/dt = a21 * m + a22 * p + b21 * a
a21 = C**2 * (m_max - m_min) / (p_max - p_min) / Vp / 1e6
a22 = -C**2 / Vp * denormalize(a_exp,'a') * kv * 500 / np.sqrt((denormalize(p_exp,'p') - P_out) * 1000) / 1e6
b21 = -C**2 / (p_max - p_min) / Vp * (a_max - a_min) * kv * np.sqrt((denormalize(p_exp,'p') - P_out) * 1000) / 1e6

Ac = np.array([[a11, a12],
               [a21, a22]])
Bc = np.array([[0, b12 ],
               [b21, 0]])
Cc = np.eye(2)
Dc = np.zeros((2,2))

Ts = 0.5

sys_c = ctrl.ss(Ac, Bc, Cc, Dc)
sys_d = ctrl.c2d(sys_c, Ts, method='zoh')

# MPC problem setup
n = 10
nx = 2 * n
nu = 2 * n
scaling = 1 * 1e0

A, B, C, D = ctrl.ssdata(sys_d)
N = 2

Rtilde = np.array([[10,1], [10,1]])
Ptilde = np.array([[10,10], [10,10]])
Qtilde = np.array([[10,10], [10,10]])

R = Rtilde
P = Ptilde
for j in range(n-1):
    R = block_diag(R, Rtilde)
    P = block_diag(Qtilde, P)

My = np.zeros((N * n, N * n))
Gtilde = np.array([[normalize(0.65,'a'), normalize(5,'n')], [normalize(0.35,'a'), normalize(2.7,'n')]])
G = Gtilde
for i in range(n):
    for j in range(i+1):
        if (j+1)*N <= N* n:
            My[i*N:(i+1)*N, j*N:(j+1)*N] = np.linalg.matrix_power(A, i-j) @ B
    if i < n-1:
        G = block_diag(G, Gtilde)

H = R + My.T @ P @ My
F_add_quadprog = 1 * np.hstack([(A.T @ Qtilde @ B).reshape(-1,1), np.zeros((N*N, N*n-1))]).T
F_add = F_add_quadprog * 1
S = G @ np.linalg.inv(H) @ F_add

w = np.ones((2 * n, 1)) * scaling

D = np.eye(nx) - G @ np.linalg.inv(H) @ G.T

b_qp = w
A_qp = G

# Simulate the MPC policy
K = 25

x0 = -np.array([[1], [1]]) * 2e2
xk = x0.copy()
u_mpc = np.zeros(K)
u_YALMIP = np.zeros((n, K))
xk_store_MPC = np.zeros((N, K))

for j in range(K):
    u = cp.Variable((n, 1))
    f_yalmip = xk.T @ (2 * np.hstack([(A.T @ Qtilde @ B).reshape(-1,1), np.zeros((N, n-1))]))
    cost = cp.quad_form(u, H) + 1e0 * f_yalmip @ u
    constraints = [A_qp @ u <= b_qp]
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()
    
    u_YALMIP[:, j:j+1] = u.value
    
    u_mpc[j] = u_YALMIP[0, j]
    xk = A @ xk + B * u_mpc[j]
    xk_store_MPC[:, j:j+1] = xk

# Simulate the implicit neural network policy
iters = 1000  # number of iterations of the NN unraveling
y_store = np.zeros((nx, iters))
phi_store = np.zeros((nx, iters))
res_store = np.zeros((nx, iters))
res_norm = np.zeros(iters)
sign_store = np.ones((nx, iters))

xk = x0.copy()
c_MPC = 1 * S @ xk + 1 * w  # initialize the implicit NN
zeta = -1 * c_MPC

y0 = np.zeros((nx, 1))
phi = np.zeros((nx, 1))
for g in range(nx):
    if y0[g] >= 0:
        phi[g] = y0[g]
    else:
        phi[g] = 0

residual = y0 - D @ phi - zeta
K_gain = 0
y = y0.copy()

u_nn = np.zeros(K)
u_ramp = np.zeros((n, K))
xk_store_nn = np.zeros((N, K))
res_norm_store = np.zeros((K, iters))

for k in range(K):
    for j in range(iters):  # unravel the implicit NN into an explicit one
        c_MPC = 1 * S @ xk + 1 * w  # define the implicit NN weights
        zeta = -1 * c_MPC
        y_store[:, j:j+1] = y
        phi_store[:, j:j+1] = phi
        res_store[:, j:j+1] = residual
        ykp1 = D @ phi + zeta + 1 * K_gain * residual
        
        phi = np.zeros((nx, 1))
        for g in range(nx):
            if ykp1[g] > 0:
                phi[g] = ykp1[g]
            else:
                phi[g] = 0
                sign_store[g, j] = -1
        
        y = ykp1.copy()
        residual = y - D @ phi - zeta  # compute residuals
        res_norm[j] = np.linalg.norm(residual)
        res_norm_store[k, j] = np.linalg.norm(y - D @ phi - zeta)
    
    y_ramp = ykp1.copy()
    u_ramp[:, k:k+1] = -np.linalg.inv(H) @ (F_add @ xk + G.T @ phi)  # get the output
    
    u_nn[k] = u_ramp[0, k]  # the step-ahead control action
    xk = A @ xk + B * u_nn[k]  # implement the state update
    xk_store_nn[:, k:k+1] = xk

combined = np.vstack([xk_store_nn, xk_store_MPC])  # compare the results

# Plot the results
plt.rcParams.update({'font.size': 12})

# Figure 1: x1 state
plt.figure(figsize=(8, 5))
plt.plot(range(K), xk_store_nn[0, :], '-xk', color=[0.2, 0.2, 0.2], linewidth=2, markersize=8)
plt.plot(range(K), xk_store_MPC[0, :], '+k', color=[0.6, 0.6, 0.6], linewidth=2, markersize=8)
plt.grid(True)
plt.xlabel('Time instant $k$')
plt.ylabel('$x_1[k]$')
plt.legend(['Implicit NN controller', 'MPC Controller'], loc='lower left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'x1_state.png'), dpi=300)
plt.close()

# Figure 2: x2 state
plt.figure(figsize=(8, 5))
plt.plot(range(K), xk_store_nn[1, :], '-xk', color=[0.2, 0.2, 0.2], linewidth=2, markersize=8)
plt.plot(range(K), xk_store_MPC[1, :], '+k', color=[0.6, 0.6, 0.6], linewidth=2, markersize=8)
plt.grid(True)
plt.xlabel('Time instant $k$')
plt.ylabel('$x_2[k]$')
plt.legend(['Implicit NN controller', 'MPC Controller'], loc='lower left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'x2_state.png'), dpi=300)
plt.close()

# Figure 3: Control input
plt.figure(figsize=(8, 5))
plt.plot(range(K), u_ramp[0, :], '-xk', color=[0.2, 0.2, 0.2], linewidth=2, markersize=8)
plt.plot(range(K), u_mpc, '+k', color=[0.6, 0.6, 0.6], linewidth=2, markersize=8)
plt.grid(True)
plt.xlabel('Time instant $k$')
plt.ylabel('Input $u[k]$')
plt.legend(['Implicit NN controller', 'MPC Controller'], loc='lower left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'control_input.png'), dpi=300)
plt.close()

# Figure 4: Residual norms for first few time steps
plt.figure(figsize=(8, 5))
set_K = 7
for j in range(set_K):
    plt.plot(range(iters), res_norm_store[j, :], '-k', color=[j/(set_K+1)]*3, linewidth=2)
plt.grid(True)
plt.xlabel('Layer depth $j$')
plt.ylabel('Residual $\|w[j]-D\phi(w[j])-\zeta\|_2$')
plt.legend(['$k = 1$', '$k = 2$', '$k= 3$', '$k= 4$', '$k = 5$', '$k = 6$', '$k = 7$'], loc='upper left')
plt.xlim([0, 600])
plt.ylim([-0.5, 6])
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'residual_norms.png'), dpi=300)
plt.close()

# Figure 5: Surface plot of residuals
X, Y = np.meshgrid(range(iters), range(K))
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, res_norm_store, cmap='viridis', edgecolor='none')
ax.set_xlabel('Layer depth $j$')
ax.set_ylabel('Time instant $k$')
ax.set_zlabel('Residual norm')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'residual_surface.png'), dpi=300)
plt.close()

plt.show()