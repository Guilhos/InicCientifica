import numpy as np
import casadi as ca
from scipy.optimize import fsolve

class Simulation:
    def __init__(self,p,m,steps,dt=0.5):
        self.A1 = (2.6)*(10**-3)
        self.Lc = 2
        self.kv = 0.38
        self.P1 = 4.5
        self.P_out = 5
        self.C = 479
        self.alphas = [0.5]*m
        self.N_RotS = [38500]*m
        self.dt = dt
        self.y = []
        self.u = []
        self.p = p
        self.m = m
        self.steps = steps

        try:
            # Tentando importar de libs.Interpolation
            from libs.Interpolation import Interpolation
        except ImportError:
            try:
                # Tentando importar diretamente de Interpolation
                from Interpolation import Interpolation
            except ImportError:
                print("Falha ao importar 'Interpolation' de ambos os caminhos.")

        interp = Interpolation('NNMPC/libs/tabela_phi.csv')
        interp.load_data()
        self.lut = interp.interpolate()      

    def fun(self, variables, alpha, N, lut):
        (x, y) = variables  # x e y são escalares
        phi_value = float(lut([N, x]))  # Garantir que phi_value é escalar
        eqn_1 = (self.A1 / self.Lc) * ((phi_value * self.P1) - y) * 1e3
        eqn_2 = (self.C**2) / 2 * (x - alpha * self.kv * np.sqrt(y * 1000 - self.P_out * 1000))
        return [eqn_1, eqn_2]

    def pIniciais(self):
        # Condições iniciais
        result = fsolve(self.fun, (10, 10), args=(self.alphas[0],self.N_RotS[0],self.lut))
        init_m, init_p = result

        # Variáveis CasADi
        x = ca.MX.sym('x', 2)
        p = ca.MX.sym('p', 2)  # Parâmetros (alpha e N)
        alpha, N = p[0], p[1]  # Divisão dos parâmetros

        # Solução Numérica
        rhs = ca.vertcat((self.A1 / self.Lc) * ((self.lut(ca.vertcat(N, x[0])) * self.P1) - x[1]) * 1e3,
                         (self.C**2) / 2 * (x[0] - alpha * self.kv * np.sqrt(x[1] * 1000 - self.P_out * 1000)))

        ode = {'x': x, 'ode': rhs, 'p': p}

        F = ca.integrator('F', 'cvodes', ode, 0,self.dt)

        for j in range(self.p):
            if j < self.m:
                params = [self.alphas[j-1], self.N_RotS[j-1]]
            sol = F(x0=[init_m, init_p], p=params)
            xf_values = np.array(sol["xf"])
            aux1, aux2 = xf_values
            init_m = aux1[-1]
            init_p = aux2[-1]
            self.y.append([aux1[0], aux2[0]])
            if j < self.m:
                self.u.append([self.alphas[j], self.N_RotS[j]])
        
        y0 = np.array(self.y[:self.steps]).reshape(-1,1)
        u0 = np.array(self.u[:self.steps]).reshape(-1,1)
        
        return (y0, u0)
    
    def pPlanta(self, y0, dU):
        self.y = []
        self.alphas.append(self.alphas[-1] + float(dU[0]))
        self.alphas = self.alphas[1:]
        self.N_RotS.append(self.N_RotS[-1] + float(dU[1]))
        self.N_RotS = self.N_RotS[1:]
        init_m = y0[-2].item()
        init_p = y0[-1].item()

         # Variáveis CasADi
        x = ca.MX.sym('x', 2)
        p = ca.MX.sym('p', 2)  # Parâmetros (alpha e N)
        alpha, N = p[0], p[1]  # Divisão dos parâmetros

        # Solução Numérica
        rhs = ca.vertcat((self.A1 / self.Lc) * ((self.lut(ca.vertcat(N, x[0])) * self.P1) - x[1]) * 1e3,
                         (self.C**2) / 2 * (x[0] - alpha * self.kv * np.sqrt(x[1] * 1000 - self.P_out * 1000)))

        ode = {'x': x, 'ode': rhs, 'p': p}

        F = ca.integrator('F', 'cvodes', ode, 0,self.dt)

        for j in range(self.p):
            if j < self.m:
                params = [self.alphas[j-1], self.N_RotS[j-1]]
            sol = F(x0=[init_m, init_p], p=params)
            xf_values = np.array(sol["xf"])
            aux1, aux2 = xf_values
            init_m = aux1[-1]
            init_p = aux2[-1]
            self.y.append([aux1[0], aux2[0]])
            if j < self.m:
                self.u.append([self.alphas[j], self.N_RotS[j]])

        self.y = np.array(self.y).reshape(-1,1)
        self.uk = np.array(self.u).reshape(-1,1)[-2:]
        print(self.uk)
        return self.y, self.uk
    
    def ySetPoint(self, nSP):
        SPlist = []
        for i in range(nSP):
            result = fsolve(self.fun, (10, 10), args=(np.random.randint(35,65)/100,np.random.randint(27e3,5e4),self.lut))
            SPlist.append(result)
        return SPlist


if __name__ == '__main__':

    dU = [0,0,0,0,0,0]
    sim = Simulation(1,1,3)
    y0, u0 = sim.pIniciais()

    yPlanta = sim.pPlanta(y0, dU)
    
    SPlist = sim.ySetPoint(3)
    print(SPlist)



