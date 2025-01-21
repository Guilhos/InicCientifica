import numpy as np
from scipy.optimize import fsolve
import casadi as ca
import matplotlib.pyplot as plt
import time

class Simulation:
    def __init__(self, A1, Lc, kv, P1, P_out, C, alphas, nCiclos, nData, perturb,tempo, dt):
        self.A1 = A1
        self.Lc = Lc
        self.kv = kv
        self.P1 = P1
        self.P_out = P_out
        self.C = C
        self.alphas = alphas
        self.nCiclos = nCiclos
        self.nData = nData
        self.perturb = perturb
        self.dt = dt
        self.tempo = tempo

        self.interval = [np.linspace(i * self.tempo, (i + 1) * self.tempo, self.nData) for i in range(self.nCiclos)]
        self.time = 0
        
        self.alpha_values = []
        self.massFlowrate = []
        self.PlenumPressure = []
        self.Phi_values = []
        self.RNN_train = []
        self.RNN_trainFut = []
        

    def fun(self, variables, alpha):
        (x, y) = variables  # x e y são escalares
        eqn_1 = (self.A1 / self.Lc) * ((1.5 * self.P1) - y) * 1e3
        eqn_2 = (self.C**2) / 2 * (x - alpha * self.kv * np.sqrt(y * 1000 - self.P_out * 1000))
        return [eqn_1, eqn_2]


    def run(self):
        # Condições iniciais
        result = fsolve(self.fun, (10, 10), args=(self.alphas[0]))
        print(result)
        init_m, init_p = result

        # Variáveis CasADi
        x = ca.MX.sym('x', 2)
        alpha = ca.MX.sym('alpha', 1)  # Parâmetros (alpha e N)

        # Solução Numérica
        tm1 = time.time()
        for i in range(self.nCiclos):
            alpha_value = self.alphas[i] + np.random.normal(0, self.perturb, self.nData)
            self.alpha_values.append(alpha_value)

            rhs = ca.vertcat((self.A1 / self.Lc) * ((1.5 * self.P1) - x[1]) * 1e3,
                             (self.C**2) / 2 * (x[0] - alpha * self.kv * np.sqrt(x[1] * 1000 - self.P_out * 1000)))
            
            ode = {'x': x, 'ode': rhs, 'p': alpha}

            F = ca.integrator('F', 'cvodes', ode, self.interval[0][0],self.dt)

            for j in range(self.nData):
                params = [alpha_value[j]]
                sol = F(x0=[init_m, init_p], p=params)
                xf_values = np.array(sol["xf"])
                aux1, aux2 = xf_values
                self.massFlowrate.append(aux1)
                self.PlenumPressure.append(aux2)
                init_m = aux1[-1]
                init_p = aux2[-1]
                self.RNN_train.append([aux1[0], aux2[0], alpha_value[j]])
                self.RNN_trainFut.append([aux1[0], aux2[0], alpha_value[j]])

        tm2 = time.time()
        self.time = tm2-tm1
        self.massFlowrate = np.reshape(self.massFlowrate, [self.nCiclos, self.nData])
        self.PlenumPressure = np.reshape(self.PlenumPressure, [self.nCiclos, self.nData]) 

np.random.seed(42)
print(np.random.seed)

# Constantes
A1 = (2.6)*(10**-3)
Lc = 2
kv = 0.38
P1 = 4.5
P_out = 5
C = 479

timestep = 3 # Passos no passado para prever o próximo
nCiclos = 7 # Número de vezes que o Alfa irá mudar, considere o treino e os testes.
alphas = np.random.uniform(0.35,0.65, nCiclos+1) # Abertura da válvula
epochs = 1500
nData = 600
nDataTeste = nData//nCiclos
perturb = 5e-3
tempo = 60
dt = 0.1 # Tempo amostral

# Variáveis auxiliares
interval = [np.linspace(i * tempo, (i + 1) * tempo, nData) for i in range(nCiclos)]
interval_test = [np.linspace(i * tempo, (i + 1) * tempo, nDataTeste) for i in range(nCiclos)]
massFlowrate = []
PlenumPressure = []
alpha_values = []
RNN_train = []
RNN_trainFut = []

sim = Simulation(A1, Lc, kv, P1, P_out, C, alphas, nCiclos, nData, perturb, tempo, dt)
# Execute a simulação
sim.run()

RNN_train = sim.RNN_train
RNN_trainFut = sim.RNN_trainFut
massFlowrate = sim.massFlowrate
PlenumPressure = sim.PlenumPressure
alpha_values = sim.alpha_values

print(sim.time)

plt.rcParams.update({
    'font.size': 22,  # Aumenta o tamanho da fonte geral
    'axes.titlesize': 24,  # Tamanho do título dos eixos
    'axes.labelsize': 32,  # Tamanho dos rótulos dos eixos
    'xtick.labelsize': 25,  # Tamanho dos rótulos do eixo X
    'ytick.labelsize': 25,  # Tamanho dos rótulos do eixo Y
    'legend.fontsize': 25,  # Tamanho da fonte da legenda
})

# Plot 2: Pressão vs Tempo
plt.figure(figsize=(16, 9))
for i in range(nCiclos):
    plt.plot(interval[i], np.squeeze(PlenumPressure[i]), label=f'Ciclo {i + 1}', 
             linewidth=4, color='blue')  # Todas as linhas em azul
plt.xlabel("Tempo / s")
plt.ylabel("Pressão / MPa")
plt.grid(True)
plt.show()
