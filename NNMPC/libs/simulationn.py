import numpy as np
import casadi as ca
import time
from scipy.optimize import fsolve
import torch

class Simulation:
    def __init__(self,  interpolation, dt=0.1):
        self.A1 = (2.6)*(10**-3)
        self.Lc = 2
        self.kv = 0.38
        self.P1 = 4.5
        self.P_out = 5
        self.C = 479
        self.alphas = np.random.uniform(0.35,0.65,3)
        self.N_RotS = np.random.uniform(27e3, 5e4,3)
        self.dt = dt
        
        #Interpolação
        self.interpolation = interpolation
        self.data = None
        self.N_rot = None
        self.Mass = None
        self.Phi = None

        self.time = 0
        
        self.alpha_values = []
        self.N_values = []
        self.massFlowrate = []
        self.PlenumPressure = []
        self.Phi_values = []
        self.output = []
        self.RNN_trainFut = []
        self.X_train = []
        self.y_train = []        

    def fun(self, variables, alpha, N):
        (x, y) = variables  # x e y são escalares
        phi_value = float(self.interpolation([N, x]))  # Garantir que phi_value é escalar
        eqn_1 = (self.A1 / self.Lc) * ((phi_value * self.P1) - y) * 1e3
        eqn_2 = (self.C**2) / 2 * (x - alpha * self.kv * np.sqrt(y * 1000 - self.P_out * 1000))
        return [eqn_1, eqn_2]

    def run(self):
        lut = self.interpolation
        # Condições iniciais
        result = fsolve(self.fun, (10, 10), args=(self.alphas[0],self.N_RotS[0]))
        init_m, init_p = result

        # Variáveis CasADi
        x = ca.MX.sym('x', 2)
        p = ca.MX.sym('p', 2)  # Parâmetros (alpha e N)
        alpha, N = p[0], p[1]  # Divisão dos parâmetros

        # Solução Numérica
        tm1 = time.time()

        rhs = ca.vertcat((self.A1 / self.Lc) * ((lut(ca.vertcat(N, x[0])) * self.P1) - x[1]) * 1e3,
                         (self.C**2) / 2 * (x[0] - alpha * self.kv * np.sqrt(x[1] * 1000 - self.P_out * 1000)))

        ode = {'x': x, 'ode': rhs, 'p': p}

        F = ca.integrator('F', 'cvodes', ode, 0,self.dt)

        for j in range(3):
            params = [self.alphas[j], self.N_RotS[j]]
            sol = F(x0=[init_m, init_p], p=params)
            xf_values = np.array(sol["xf"])
            aux1, aux2 = xf_values
            self.massFlowrate.append(aux1)
            self.PlenumPressure.append(aux2)
            init_m = aux1[-1]
            init_p = aux2[-1]
            self.output.append([aux1[0], aux2[0], self.alphas[j], self.N_RotS[j]])

        tm2 = time.time()
        self.time = tm2-tm1
        
        self.output = np.array(self.output)

if __name__ == '__main__':
    from Interpolation import Interpolation

    lut = Interpolation('./../tabela_phi.csv')
    lut.load_data()
    interpolation = lut.interpolate()

    sim = Simulation(interpolation)
    sim.run()
    print(sim.output)
