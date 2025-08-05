import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.linalg import block_diag
import os

output_dir = "MPC_to_NN/MPCvsNN - Drummond/output_figures"
os.makedirs(output_dir, exist_ok=True)

# MPC problem setup
n = 10
nx = 2 * n
scaling = 1 * 1e0

A = np.array([[4/3, -2/3], [1, 0]])
B = np.array([[0], [1]])
N = 2

Rtilde = 1
Ptilde = np.array([[7.1667, -4.2222], [-4.2222, 4.6852]])
Qtilde = np.array([[1, -2/3], [-2/3, 3/22]])

R = Rtilde
P = Ptilde
for j in range(n-1):
    R = block_diag(R, Rtilde)
    P = block_diag(Qtilde, P)

My = np.zeros((N * n, n))
Gtilde = np.array([[0.1], [-0.1]])
G = Gtilde
for i in range(n):
    for j in range(i+1):
        My[i*N:(i+1)*N, j] = (np.linalg.matrix_power(A, i-j) @ B).flatten()
    if i < n-1:
        G = block_diag(G, Gtilde)

H = R + My.T @ P @ My
F_add_quadprog = 1 * np.hstack([(A.T @ Qtilde @ B).reshape(-1,1), np.zeros((N, n-1))]).T
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