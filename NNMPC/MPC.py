import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
from libs.simulationn import Simulation
from NN_Model import NN_Model

class PINN_MPC():
    def __init__(self, p, m, q, r, timesteps):
        #Constantes
        self.p = p
        self.m = m
        self.q = q
        self.r = r
        self.timesteps = timesteps
        self.nU = 2
        self.nY = 2

        # dU inicial
        self.dU = np.zeros((self.nU * self.m, 1))
        self.dU = np.concatenate((np.array(self.dU), np.zeros((self.nU * (p-m), 1))))

        # Objetos
        self.sim = Simulation()
        self.NNMod = NN_Model(p,m)

        # Limites das variáveis
        self.u_min = np.array([[0.35], [27e3]])
        self.u_max = np.array([[0.65], [5e4]])
        self.dU_min = np.array([[0.01], [500]])
        self.dU_max = np.array([[0.15], [5000]])
        self.y_min = np.array([[3.5], [5.27]])
        self.y_max = np.array([[12.3], [10.33]])

        # Setpoint provisório
        self.y_sp = (self.y_max + self.y_min) / 2
    
    # Pontos, Pontos Iniciais, Planta, Modelo

    def pPlanta(self, dU):
        y0, u0, yPlanta = self.sim.run(self.p, self.m, dU)
        return y0, u0, yPlanta
    
    def pModelo(self, y0, u0, dU):
        yModel, uModel = self.NNMod.run(y0,u0,dU)
        return yModel, uModel
    
    # Ajuste das Matrizes

    def iTil(self, n, x):
        n = np.tile(n, (x,1))
        return n
    
    def diagMatrix(self, x,n):
        x = np.float64(x)
        n = int(n)
        X_matrix = np.full((n,n),0, dtype=np.float64)
        np.fill_diagonal(X_matrix,x)
        return X_matrix

    def ajusteMatrizes(self):
        self.y_sp = ca.DM(self.iTil(self.y_sp,self.p).reshape(-1,1)) # Expansão do y_setpoint para P. SHAPE -> (nY*P, 1)
        self.y_min = ca.DM(self.iTil(self.y_min,self.p)) # Expansão do y_min para P. SHAPE -> (nY*P, 1)
        self.y_max = ca.DM(self.iTil(self.y_max,self.p)) # Expansão do y_max para P. SHAPE -> (nY*P, 1)

        self.u_min = ca.DM(self.iTil(self.u_min,self.p)) # Expansão do u_min para M. SHAPE -> (nU*M, 1)
        self.u_max = ca.DM(self.iTil(self.u_max,self.p)) # Expansão do u_max para M. SHAPE -> (nU*M, 1)

        self.dU_min = self.iTil(self.dU_min, self.m) # Expansão do dU_min para M. SHAPE -> (nU*M, 1)
        self.dU_min = ca.DM(np.concatenate((self.dU_min,np.zeros((int(self.nU) * (self.p - self.m), 1))))) # Adição de P - M linhas de 0. SHAPE -> (nU*P, 1)
        self.dU_max = self.iTil(self.dU_max, self.m) # Expansão do dU_max para M. SHAPE -> (nU*M, 1)
        self.dU_max = ca.DM(np.concatenate((self.dU_max,np.zeros((int(self.nU) * (self.p - self.m), 1))))) # Adição de P - M linhas de 0. SHAPE -> (nU*P, 1)

        self.q = ca.DM(self.diagMatrix(self.q,self.nY*self.p)) # Criação de uma matriz com os valores de Q na diagonal. SHAPE -> (nY*p, nY*p)
        self.r = ca.DM(self.diagMatrix(self.r,self.nU*self.p)) # Criação de uma matriz com os valores de R na diagonal. SHAPE -> (nU*p, nU*p)

    # Otimização

    def otimizar(self, yModel, uModel, yPlanta):
        yModel = ca.DM(yModel)
        yPlanta = ca.DM(yPlanta)
        dY = yModel - yPlanta
        
        Fs = ca.MX.sym('f')
        dUs = ca.MX.sym('dU',self.nU * self.p, 1)

        x = ca.vertcat(dUs, Fs)

        g = ca.vertcat(yModel, uModel)

        dU_init = ca.DM(self.dU)
        x0 = ca.vertcat(dU_init, (yModel - self.y_sp + dY).T @ self.q @ (yModel - self.y_sp + dY) + dU_init.T @ self.r @ dU_init)

        x_min = ca.vertcat(self.dU_min, (yModel - self.y_sp + dY).T @ self.q @ (yModel - self.y_sp + dY) + self.dU_min.T @ self.r @ self.dU_min)
        x_max = ca.vertcat(self.dU_max, (yModel - self.y_sp + dY).T @ self.q @ (yModel - self.y_sp + dY) + self.dU_max.T @ self.r @ self.dU_max)

        # Garantindo que lbg e ubg estejam no formato correto
        lbg = ca.vertcat(self.y_min, self.u_min - dU_init)  # Transformar em array antes do DM
        ubg = ca.vertcat(self.y_max, self.u_max - dU_init)

        nlp = {'x': x, 'f': Fs, 'g': g}
        solver = ca.nlpsol('solver', 'ipopt', nlp)

        sol = solver(x0=x0, lbg = lbg, ubg = ubg, lbx = x_min, ubx = x_max)
        # Extraindo os resultados
        dU_opt = sol['x'].full().flatten()
        return dU_opt
    
    def run(self):
        self.ajusteMatrizes()
        for i in range(2):
            y0, u0, yPlanta = self.pPlanta(self.dU)
            yModel, uModel = self.pModelo(y0, u0, self.dU)
            dU_opt = self.otimizar(yModel, uModel, yPlanta)
            dU_opt = dU_opt[:-1].reshape((100,1))
            self.dU = dU_opt

            print(yModel,uModel)
        return dU_opt

if __name__ == '__main__':
    p, m, q, r, timesteps = 50, 3, 0.1, 1, 3
    mpc = PINN_MPC(p, m, q, r, timesteps)
    dU_opt = mpc.run()
    print("Controle ótimo:", dU_opt, dU_opt.shape)
