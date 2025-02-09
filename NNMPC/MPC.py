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
        self.sim = Simulation(p,m)
        self.NNMod = NN_Model(p,m)

        # Limites das variáveis
        self.u_min = np.array([[0.35], [27e3]])
        self.u_max = np.array([[0.65], [5e4]])
        self.dU_min = np.array([[0.01], [500]])
        self.dU_max = np.array([[0.15], [5000]])
        self.y_min = np.array([[3.5], [5.27]])
        self.y_max = np.array([[12.3], [10.33]])

        # Setpoint provisório
        self.y_sp = np.array([[10], [8]])
    
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

        x_min = ca.vertcat(self.dU_min, (yModel - self.y_sp + dY).T @ self.q @ (yModel - self.y_sp + dY) + dU_init.T @ self.r @ dU_init)
        x_max = ca.vertcat(self.dU_max, (yModel - self.y_sp + dY).T @ self.q @ (yModel - self.y_sp + dY) + dU_init.T @ self.r @ dU_init)

        lbg = ca.vertcat(self.y_min, self.u_min) 
        ubg = ca.vertcat(self.y_max, self.u_max)

        nlp = {'x': x, 'f': Fs, 'g': g}
        options = {'print_time': False}
        solver = ca.nlpsol('solver', 'ipopt', nlp, options)

        sol = solver(x0=x0, lbg = lbg, ubg = ubg, lbx = x_min, ubx = x_max)
        # Extraindo os resultados
        dU_opt = sol['x'].full().flatten()
        return dU_opt
    
    def run(self):
        self.ajusteMatrizes()
        y0, u0 = self.sim.pIniciais()
        y0p = y0
        resM = []
        resP = []
        iter = 3
        for i in range(iter):
            print(y0,u0)
            yPlanta = self.sim.pPlanta(y0p, self.dU)
            yModel, uModel = self.NNMod.run(y0,u0,self.dU)
            resM.append(yModel[::2])
            resP.append(yModel[1::2])

            dU_opt = self.otimizar(yModel, uModel, yPlanta)
            dU_opt = dU_opt[:-1].reshape((100,1))
            self.dU = dU_opt
            y0 = yModel[-self.timesteps*self.nY:]
            u0 = uModel[-self.timesteps*self.nU:]
            y0p = yPlanta[-self.timesteps*self.nY:]

            print(dU_opt[:6])

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

        x = np.linspace(0,iter*self.p,iter*self.p)
        y_spM = np.full_like(x, self.y_sp[0])
        y_spP = np.full_like(x, self.y_sp[1])
        axes[0].plot(x, np.array(resM).reshape(iter * p, 1), label="resM")
        axes[0].plot(x, y_spM, linestyle="--", color="red", label="y_sp")
        axes[0].set_title("resM")
        axes[0].set_ylabel("Valor")
        axes[0].legend()
        axes[0].grid()

        axes[1].plot(x, np.array(resP).reshape(iter * p, 1), label="resP", color="green")
        axes[1].plot(x, y_spP, linestyle="--", color="red", label="y_sp")
        axes[1].set_title("resP")
        axes[1].legend()
        axes[1].grid()

        plt.xlabel("Tempo")
        plt.suptitle("Comparação de resM e resP")
        plt.tight_layout()

        plt.show()

        return dU_opt

if __name__ == '__main__':
    p, m, q, r, timesteps = 50, 3, 0.1, 1, 3
    mpc = PINN_MPC(p, m, q, r, timesteps)
    dU_opt = mpc.run()
    print("Controle ótimo:", dU_opt, dU_opt.shape)
