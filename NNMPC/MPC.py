import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from libs.simulationn import Simulation
from libs.Interpolation import Interpolation
from NN_Model import NN_Model

lut = Interpolation('libs/tabela_phi.csv')
lut.load_data()
interpolation = lut.interpolate()

# Definições do problema
N = 10  # Horizonte de predição
dt = 0.1  # Intervalo de tempo
n_states = 1  # Número de estados
n_controls = 1  # Número de controles

# Limites das variáveis
u_min, u_max = -2, 2  # Limites na variável manipulada
delta_u_min, delta_u_max = -0.5, 0.5  # Limites nos incrementos de u
y_min, y_max = -1, 1  # Limites na variável controlada

# Parâmetros
p = 1.0  # Parâmetro da função F(u, p)

# Modelo do sistema
def system_dynamics(x, u, p):
    """
    Dinâmica do sistema.
    x: estado atual
    u: entrada manipulada
    p: parâmetros
    """
    return x + u * p * dt

# Função de custo
def cost_function(U, x0, y_ref, p):
    """
    Calcula o custo total ao longo do horizonte de predição.
    """
    x = x0
    cost = 0
    for k in range(N):
        u = U[k]
        x = system_dynamics(x, u, p)
        cost += (x - y_ref) ** 2  # Erro quadrático em relação à referência
    return cost

# Restrições (controladas, manipuladas e incrementos)
def constraint(U, x0, p):
    """
    Define todas as restrições do problema.
    """
    x = x0
    constraints = []
    for k in range(N):
        u = U[k]
        x = system_dynamics(x, u, p)
        
        # Restrições nas variáveis controladas
        constraints.append(y_min - x)  # y >= y_min
        constraints.append(x - y_max)  # y <= y_max
        
        # Restrições nas variáveis manipuladas
        constraints.append(u_min - u)  # u >= u_min
        constraints.append(u - u_max)  # u <= u_max
        
        # Restrições nos incrementos
        if k > 0:
            delta_u = U[k] - U[k - 1]
            constraints.append(delta_u_min - delta_u)  # Delta u >= delta_u_min
            constraints.append(delta_u - delta_u_max)  # Delta u <= delta_u_max
    return np.array(constraints)

# Estado inicial e referência desejada
x0 = 0  # Estado inicial
y_ref = 0.5  # Referência da saída

# Chute inicial para U
U0 = np.zeros(N)

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
