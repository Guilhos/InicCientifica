import casadi as ca
import numpy as np

p = 12
m = 3
Nx = 2
Nu = 2

A = ca.DM([[-0.007663036261364376,	-5.681806376217251],
            [0.0005013997449098692,	0.3717659907243302]])
B = ca.DM([[8.402698186825358,	9.845415222347835],
            [-0.5706246815351778,	0.7688374654297399]])
C = ca.DM([[1, 0],
            [0, 1]])
D = ca.DM([[0, 0],
            [0, 0]])

A_ = ca.blockcat([[A, B],
                    [ca.DM.zeros(Nu, Nu), ca.DM.eye(Nu)]])
B_ = ca.blockcat([[B], [ca.DM.eye(Nu)]])

Qtil = ca.DM([[2, 0],[0, 2]])
Rtil = ca.DM([[2, 0], [0, 2]])

Q = ca.diagcat(*[Qtil for _ in range(p)])
R = ca.diagcat(*[Rtil for _ in range(m)])

C_ = ca.blockcat([[C, ca.DM.zeros(Nu, Nu)]])
Cu = ca.blockcat([[ca.DM.zeros(Nx, Nx), ca.DM.eye(Nu)]])

psi = C_ @ A_
for i in range(1, p):
    psi = ca.vertcat(psi, psi[-Nx:,:] @ A_)

theta = ca.DM.zeros((Nx*p, Nu*m))
for i in range(p):
    for j in range(i+1):
        if (j+1)*Nu <= Nu*m:
            theta[i*Nu:(i+1)*Nu, j*Nu:(j+1)*Nu] = C_ @ ca.mpower(A_, i-j) @ B_
        else:
            continue

psiu = Cu @ A_
for i in range(1, p):
    psiu = ca.vertcat(psiu, psiu[-Nx:,:] @ A_)

thetau = ca.DM.zeros((Nx*p, Nu*m))
for i in range(p):
    for j in range(i+1):
        if (j+1)*Nu <= Nu*m:
            thetau[i*Nu:(i+1)*Nu, j*Nu:(j+1)*Nu] = Cu @ ca.mpower(A_, i-j) @ B_
        else:
            continue

H =  theta.T @ Q @ theta + R
H = (H + H.T)/2
F = ca.vertcat(psi.T @ Q @ theta,
                ca.kron(ca.DM.ones(p,1),ca.DM.eye(Nx)).T @ -Q @ theta)

G = ca.vertcat(theta,
                -theta,
                thetau,
                -thetau,
                ca.DM.eye(Nx*p)[:, :Nu*m],
                -ca.DM.eye(Nx*p)[:, :Nu*m])


invH = ca.inv(H)
D = ca.DM.eye(Nx*p*6) - G @ invH @ G.T
D = (D + D.T)/2
D_np = np.array(D)          # converte para NumPy
D_norm = np.linalg.norm(D_np, 2)   # norma espectral
print(D_norm)
