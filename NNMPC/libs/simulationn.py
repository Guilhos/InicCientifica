import numpy as np
import casadi as ca
from scipy.optimize import fsolve

class Simulation:
    def __init__(self,p,m,steps,dt=0.5, caller = None):
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
        self.ca_YPredFun, self.ca_UPredFun = self.ca_Pred_Function()      

    def fun(self, variables, alpha, N, lut):
        (x, y) = variables  # x e y são escalares
        phi_value = float(lut([N, x]))  # Garantir que phi_value é escalar
        eqn_1 = (self.A1 / self.Lc) * ((phi_value * self.P1) - y) * 1e3
        eqn_2 = (self.C**2) / 2 * (x - alpha * self.kv * np.sqrt(y * 1000 - self.P_out * 1000))
        return [eqn_1, eqn_2]

    def pIniciais(self):
        # Condições iniciais
        self.alphas = [0.5]*self.m
        self.N_RotS = [38500]*self.m
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
    
    def pPlanta(self, y0, dU, caller = None):
        self.y = []
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
                self.alphas.append(self.alphas[-1] + dU[2*j])
                self.N_RotS.append(self.N_RotS[-1] + dU[2*j+1])
            params = [self.alphas[-1], self.N_RotS[-1]]
            sol = F(x0 = [init_m, init_p], p = params)
            xf_values = np.array(sol["xf"])
            aux1, aux2 = xf_values
            init_m = aux1[-1]
            init_p = aux2[-1]
            self.y.append([aux1[0], aux2[0]])
            if j < self.m:
                self.u.append([self.alphas[-1], self.N_RotS[-1]])

        self.y = np.array(self.y).reshape(-1,1)
        self.uk = np.array(self.u).reshape(-1,1)[-2:]
        print('Uk: ',self.uk)
        return self.y, self.uk
    
    def ca_Pred_Function(self):
        y0 = ca.MX.sym('y0', 6,1)
        dU = ca.MX.sym('dU', 6,1)
        alphas_init = ca.MX.sym('alphas_init', 3, 1)  # Valores iniciais de alphas
        N_RotS_init = ca.MX.sym('N_RotS_init', 3, 1)  # Valores iniciais de N_RotS
        init_m = y0[-2]
        init_p = y0[-1]
        
         # Variáveis CasADi
        x = ca.MX.sym('x', 2)
        p = ca.MX.sym('p', 2)  # Parâmetros (alpha e N)
        alpha, N = p[0], p[1]  # Divisão dos parâmetros

        # Solução Numérica
        rhs = ca.vertcat((self.A1 / self.Lc) * ((self.lut(ca.vertcat(N, x[0])) * self.P1) - x[1]) * 1e3,
                         (self.C**2) / 2 * (x[0] - alpha * self.kv * np.sqrt(x[1] * 1000 - self.P_out * 1000)))

        ode = {'x': x, 'ode': rhs, 'p': p}

        intg = ca.integrator('F', 'cvodes', ode, 0,self.dt)
        res = intg(x0 = x, p=p)
        x_next = res["xf"]
        F = ca.Function('ca_F', [x, p], [x_next])
        
        y = ca.MX()
        u = ca.MX()
        alphasMX = alphas_init  # Use os valores iniciais como simbólicos
        N_RotSMX = N_RotS_init  # Use os valores iniciais como simbólicos
        
        for j in range(self.p):
            if j < self.m:
                alphasMX = ca.vertcat(alphasMX, alphasMX[-1] + dU[2*j])
                alphasMX = alphasMX[1:]
                N_RotSMX = ca.vertcat(N_RotSMX, N_RotSMX[-1] + dU[2*j+1])
                N_RotSMX = N_RotSMX[1:]
            params = ca.vertcat(alphasMX[-1], N_RotSMX[-1])
            X_next = F(ca.vertcat(init_m, init_p), params)
            init_m = X_next[0]
            init_p = X_next[1]
            y = ca.vertcat(y, init_m, init_p)
            if j < self.m:
                u = ca.vertcat(u, alphasMX[-1], N_RotSMX[-1])
        
        Y_trend = ca.Function('ca_PredYFunction', [y0, dU, alphas_init, N_RotS_init], [y, alphasMX, N_RotSMX])
        U_trend = ca.Function('ca_PredUFunction', [y0, dU, alphas_init, N_RotS_init], [u, alphasMX, N_RotSMX])
        
        return Y_trend, U_trend
    
    def ySetPoint(self, nSP):
        SPlist = []
        for i in range(nSP):
            a = np.random.randint(45,55)/100
            b = np.random.randint(35e3,5e3)
            result = fsolve(self.fun, (12.10, 10), args=(a,b,self.lut))
            SPlist.append([result,a,b])
        return SPlist


if __name__ == '__main__':

    dU = [0.25,500,0.1,0,-0.3,700]
    sim = Simulation(50,3,3)
    y0, u0 = sim.pIniciais()

    print(sim.ySetPoint(3))



