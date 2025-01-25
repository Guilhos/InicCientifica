import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from libs.simulationn import Simulation
from NN_Model import NN_Model

'''
Ordem geral:

Nas variáveis controladas y, se segue [Vazão Volumétrica, Pressão do Plenum]
Nas variáveis manipuladas u, se segue [Abertura da válvula, Velocidade de rotação do compressor]

TODO: #1 Fazer dY funcional

'''

p = 50 # Horizonte de predição
m = 3 # Horizonte de controle
timesteps = 3
q = 0.1 # Peso Q
r =  1 # Peso R

# Conjunto de Pontos Iniciais
sim = Simulation()
y0, u0 = sim.run()
nU = len(u0) / timesteps
nY = len(y0) / timesteps

# dUs provisórios
dU = [[0.05],[2000],[-0.02],[-1000],[0.1],[1000]]
dU = np.concatenate((np.array(dU), np.zeros((int(nU) * (p-m), 1)))) # Adição de P - M linhas de 0. SHAPE -> (nY*P, 1)

# Modelo do sistema
NNModel = NN_Model(p,m)
y = NNModel.run(y0,u0,dU) # Previsão dos p próximos pontos

# Limites das variáveis
u_min, u_max = [[0.35],[27e3]], [[0.65],[5e4]]  # Limites na variável manipulada. SHAPE -> (nU, 1)
dU_min, dU_max = [[0.01], [500]], [[0.15], [5000]]  # Limites nos incrementos de u. SHAPE -> (nU, 1)
y_min, y_max = [[3.5],[5.27]], [[12.3],[10.33]] # Limites na variável controlada. SHAPE -> (nY, 1)

# Setpoint provisório
y_sp = (y_max + y_min)/2  # A média entre os valores mínimos e máximos. SHAPE -> (nY, 1)

'''Ajuste dos tamanhos das Matrizes de valores constantes'''

def iTil(n, x):
    n = np.tile(n, (x,1))
    return n

y_sp = iTil(y_sp,p) # Expansão do y_setpoint para P. SHAPE -> (nY*P, 1)
y_min = iTil(y_min,p) # Expansão do y_min para P. SHAPE -> (nY*P, 1)
y_max = iTil(y_max,p) # Expansão do y_max para P. SHAPE -> (nY*P, 1)

u_min = iTil(u_min,m) # Expansão do u_min para M. SHAPE -> (nU*M, 1)
u_max = iTil(u_max,m) # Expansão do u_max para M. SHAPE -> (nU*M, 1)

dU_min = iTil(dU_min, m) # Expansão do dU_min para M. SHAPE -> (nU*M, 1)
dU_min = np.concatenate((dU_min,np.zeros((int(nU) * (p - m), 1)))) # Adição de P - M linhas de 0. SHAPE -> (nU*P, 1)
dU_max = iTil(dU_max, m) # Expansão do dU_max para M. SHAPE -> (nU*M, 1)
dU_max = np.concatenate((dU_max,np.zeros((int(nU) * (p - m), 1)))) # Adição de P - M linhas de 0. SHAPE -> (nU*P, 1)

def diagMatrix(x,n):
    x = np.float64(x)
    X_matrix = np.full((n,n),0, dtype=np.float64)
    np.fill_diagonal(X_matrix,x)
    return X_matrix

q = diagMatrix(q,nY*p) # Criação de uma matriz com os valores de Q na diagonal. SHAPE -> (nY*p, nY*p)
r = diagMatrix(r,nU*p) # Criação de uma matriz com os valores de R na diagonal. SHAPE -> (nU*p, nU*p)

def cost_function(y, y_sp, dy, dU, q, r):
    cost = (y - y_sp + dy).T @ q @ (y - y_sp + dy) + dU.T @ r @ dU # Erro quadrático em relação à referência
    return cost

# Restrições (controladas, manipuladas e incrementos)
def constraint(u, y, p, m):
    """
    Define todas as restrições do problema.
    """
    constraints = []
    
    # Restrições nas variáveis controladas
    constraints.append(y_min - y)  # y >= y_min
    constraints.append(y - y_max)  # y <= y_max
    
    # Restrições nas variáveis manipuladas
    constraints.append(u_min - u)  # u >= u_min
    constraints.append(u - u_max)  # u <= u_max
    
    # Restrições nos incrementos
    constraints.append(dU_min - dU)  # Delta u >= delta_u_min
    constraints.append(dU - dU_max)  # Delta u <= delta_u_max

    # Restrição de f
    constraints.append(f - (y - y_sp + dy).T @ q @ (y - y_sp + dy) + dU.T @ r @ dU) # f = (y - y_sp + dy).T @ q @ (y - y_sp + dy) + dU.T @ r @ dU)

    return np.array(constraints)

# Função de otimização para passar ao solver
def optimization_function(U):
    return f

# Restrições
cons = {
    "type": "ineq",
    "fun": lambda U: constraint(U, y0, p),
}

# Resolver o problema de otimização
solution = minimize(
    optimization_function,
    u0,
    constraints=cons,
    bounds=[(dU_min, dU_max)] * p,  # Limites em U
    method="SLSQP",
    options={"disp": True},
)

# Resultados
optimal_u = solution.x
print("Valores ótimos de u ao longo do horizonte:")
print(optimal_u)
