import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from libs.simulationn import Simulation
from NN_Model import NN_Model

p = 50 # Horizonte de predição
m = 3 # Horizonte de controle
q = 0.1 # Peso Q
r =  1 # Peso R
nY = 2
nU = 2

# dUs provisórios
dU = [0.05,2000,-0.02,-1000,0.1,1000]

# Conjunto de Pontos Iniciais
sim = Simulation()
y0, u0 = sim.run()

# Modelo do sistema
NNModel = NN_Model(p,m)
y = NNModel.run(y0,u0,dU) # Previsão dos p próximos pontos


    

'''
pltz = np.linspace(1,p, p)
fig, (ax1,ax2) = plt.subplots(2,1)
ax1.plot(pltz, y[0], color='r')
ax2.plot(pltz, y[1])
plt.show()
'''

# Limites das variáveis
u_min, u_max = [0.35,27e3], [0.65,5e4]  # Limites na variável manipulada
delta_u_min, delta_u_max = [0.01, 500], [0.15, 5000]  # Limites nos incrementos de u
y_min, y_max = [3.5,5.27], [12.3,10.33] # Limites na variável controlada

# Setpoint provisório
y_sp = (y_max + y_min)/2 

def iTil(n, x):
    identidade = np.eye(n)
    iTil = np.vstack([identidade] * (n*x))
    return iTil

iTilYsp = iTil(nY,p) * y_sp


# Parâmetros
p = 1.0  # Parâmetro da função F(u, p)

def cost_function(y, y_sp, dy, dU, q, r):
    """
    Calcula o custo total ao longo do horizonte de predição.
    """
    cost = np.sum((y - y_sp + dy) ** 2) * q + np.sum(dU**2) * r # Erro quadrático em relação à referência
    return cost

# Restrições (controladas, manipuladas e incrementos)
def constraint(u, y, p, m):
    """
    Define todas as restrições do problema.
    """
    constraints = []
    
    # Restrições nas variáveis controladas
    for i in range(p):
        constraints.append(y_min - y[i])  # y >= y_min
        constraints.append(y[i] - y_max)  # y <= y_max
    
    # Restrições nas variáveis manipuladas
    for i in range(m):
        constraints.append(u_min - u[i])  # u >= u_min
        constraints.append(u[i] - u_max)  # u <= u_max
    
    # Restrições nos incrementos
    if k > 0:
        delta_u = U[k] - U[k - 1]
        constraints.append(delta_u_min - delta_u)  # Delta u >= delta_u_min
        constraints.append(delta_u - delta_u_max)  # Delta u <= delta_u_max
    return np.array(constraints)

# Função de otimização para passar ao solver
def optimization_function(U):
    return cost_function(U, x0, y_ref, p)

# Restrições
cons = {
    "type": "ineq",
    "fun": lambda U: constraint(U, x0, p),
}

# Resolver o problema de otimização
solution = minimize(
    optimization_function,
    U0,
    constraints=cons,
    bounds=[(u_min, u_max)] * N,  # Limites em U
    method="SLSQP",
    options={"disp": True},
)

# Resultados
optimal_u = solution.x
print("Valores ótimos de u ao longo do horizonte:")
print(optimal_u)
