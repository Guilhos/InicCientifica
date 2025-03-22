import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
from libs.simulationn import Simulation
from NN_Model import NN_Model
from CA_Model import CA_Model

class PINN_MPC():
    def __init__(self, p, m, q, r, steps):
        #Constantes
        self.p = p
        self.m = m
        self.q = q
        self.r = r
        self.steps = steps
        self.nU = 2
        self.nY = 2

        # dU inicial
        self.dU = np.zeros((self.nU * self.m, 1))
        #self.dU = np.concatenate((np.array(self.dU), np.zeros((self.nU * (p-m), 1))))

        # Objetos
        self.sim_pred = Simulation(p,m)
        self.sim_mf = Simulation(1,1)
        self.CAMod = CA_Model("NNMPC/libs/modelo_treinado.pth",p,m,self.nY,self.nU,self.steps)
        self.NNMod_mf = NN_Model(1,1)

        # Limites das variáveis
        self.u_min = np.array([[0.35], [27e3]])
        self.u_max = np.array([[0.65], [5e4]])
        self.dU_min = np.array([[-0.15], [-5000]])
        self.dU_max = np.array([[0.15], [5000]])
        self.y_min = np.array([[3.5], [5.27]])
        self.y_max = np.array([[12.3], [10.33]])

        # Setpoint provisório
        self.y_sp = np.array([[7.74555396], [6.66187275]])
    
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
    
    def matriz_triangular_identidade(self, m, n, N):
        matriz = np.zeros((m * N, n * N))
        
        for i in range(m):
            for j in range(n):
                if j <= i:
                    matriz[i * N:(i + 1) * N, j * N:(j + 1) * N] = np.eye(N)
        
        return ca.DM(matriz)  # Convertendo para CasADi DM

    def ajusteMatrizes(self):
        self.y_sp = ca.DM(self.iTil(self.y_sp,self.p).reshape(-1,1)) # Expansão do y_setpoint para P. SHAPE -> (nY*P, 1)
        self.y_min = ca.DM(self.iTil(self.y_min,self.p)) # Expansão do y_min para P. SHAPE -> (nY*P, 1)
        self.y_max = ca.DM(self.iTil(self.y_max,self.p)) # Expansão do y_max para P. SHAPE -> (nY*P, 1)

        self.u_min = ca.DM(self.iTil(self.u_min,self.steps)) # Expansão do u_min para M. SHAPE -> (nU*M, 1)
        self.u_max = ca.DM(self.iTil(self.u_max,self.steps)) # Expansão do u_max para M. SHAPE -> (nU*M, 1)

        self.dU_min = self.iTil(self.dU_min, self.m) # Expansão do dU_min para M. SHAPE -> (nU*M, 1)
        #self.dU_min = ca.DM(np.concatenate((self.dU_min,np.zeros((int(self.nU) * (self.p - self.m), 1))))) # Adição de P - M linhas de 0. SHAPE -> (nU*P, 1)
        self.dU_max = self.iTil(self.dU_max, self.m) # Expansão do dU_max para M. SHAPE -> (nU*M, 1)
        #self.dU_max = ca.DM(np.concatenate((self.dU_max,np.zeros((int(self.nU) * (self.p - self.m), 1))))) # Adição de P - M linhas de 0. SHAPE -> (nU*P, 1)

        self.q = ca.DM(np.diag(np.array([q[0],q[1]] * (self.nY*self.p // 2)))) # Criação de uma matriz com os valores de Q na diagonal. SHAPE -> (nY*p, nY*p)
        self.r = ca.DM(np.diag(np.array([r[0],r[1]] * (self.nU*self.m // 2)))) # Criação de uma matriz com os valores de R na diagonal. SHAPE -> (nU*p, nU*p)

    def nlp_func(self):
        yModelk = ca.MX.sym('yModelk', self.nY*self.steps, 1)
        uModelk = ca.MX.sym('uModelk', self.nU*self.steps, 1)
        yPlantak = ca.MX.sym('yPlantak', self.nY, 1)
        dUs = ca.MX.sym('dU',self.nU*self.m, 1)
        Fs = ca.MX.sym('f')

        x = ca.vertcat(dUs, Fs)
        
        dYk = yPlantak - yModelk[-self.nY:]
        print(dYk)
        dYk = ca.repmat(dYk, self.p, 1)

        dU_init = ca.MX(self.dU)
        
        yModel_pred = self.CAMod.pred_function(yModelk,uModelk,dUs)
        yModel_init = self.CAMod.pred_function(yModelk,uModelk,dU_init)
        
        matriz_inferior =  self.matriz_triangular_identidade(self.steps,self.steps,self.nU)
        
        g = ca.vertcat(yModel_pred, ca.repmat(uModelk[-2:], self.steps,1) + matriz_inferior @ dUs, Fs - (yModel_pred - self.y_sp + dYk).T @ self.q @ (yModel_pred - self.y_sp + dYk) + dU_init.T @ self.r @ dU_init)
        
        x0 = ca.vertcat(dU_init, (yModel_init - self.y_sp + dYk).T @ self.q @ (yModel_init - self.y_sp + dYk) + dU_init.T @ self.r @ dU_init)

        x_min = ca.vertcat(self.dU_min, 0)
        x_max = ca.vertcat(self.dU_max, 10e23)

        lbg = ca.vertcat(self.y_min, self.u_min, 0) 
        ubg = ca.vertcat(self.y_max, self.u_max, 0)

        return ca.Function('nlp', [yModelk, uModelk, yPlantak, dUs, Fs], [x, Fs, g, x0, x_min, x_max, lbg, ubg])

    def otimizar(self, ymk, umk, ypk):
        dUs = ca.MX.sym('dU', self.nU * self.m, 1)
        Fs = ca.MX.sym('f')

        x, Fs, g, x0, x_min, x_max, lbg, ubg = self.nlp(ymk, umk, ypk, dUs, Fs)
        nlp = {'x': x, 'f': Fs, 'g': g}
        options = {'print_time': False, 'ipopt': {'print_level': 0}}
        solver = ca.nlpsol('solver', 'ipopt', nlp, options)

        sol = solver(x0=x0, lbg=lbg, ubg=ubg, lbx=x_min, ubx=x_max)
        # Extraindo os resultados
        dU_opt = sol['x']
        return dU_opt[:-1], solver.stats()['return_status']
    
    def run(self):
        self.ajusteMatrizes()
        xmk = []
        ymk, umk = self.sim_pred.pIniciais() # Recebe os pontos iniciais, ymk [6,1] umk [6,1]
        ypk, uk = ymk[-self.nY:], umk[-self.nU:]

        self.nlp = self.nlp_func()

        Ypk = []
        Upk = []
        Ymk = []
        YspM = []
        YspP = []
        Ymink = []
        Ymaxk = []

        iter = 50
        for i in range(iter):
            print(15*'='+ f'Iteração {i+1}' + 15*'=')
            dU_opt, stats = self.otimizar(ymk, umk, ypk)
            print(stats)
            if stats == 'Solve_Succeeded':
                self.dUk = dU_opt[:self.nU]
                self.dU = ca.vertcat(dU_opt, np.zeros((self.nU, 1)))
                self.dU = self.dU[2:]
            elif stats == 'Infeasible_Problem_Detected':
                self.dUk = self.dU[:self.nU]
                self.dU = ca.vertcat(self.dU, np.zeros((self.nU, 1)))
                self.dU = self.dU[2:]
            
            umk = np.append(umk, umk[-self.nU:] + self.dUk)
            umk = umk[self.nU:]
            
            xm_2 = ca.vertcat(ymk[-6:-4],umk[-6:-4])
            xm_1 = ca.vertcat(ymk[-4:-2],umk[-4:-2])
            xmk = ca.vertcat(ymk[-2:],umk[-2:])
            
            ypk, upk = self.sim_mf.pPlanta(ypk, self.dUk)
            upk = upk.flatten()
            ypk = ypk.flatten()
            ymk_next = self.CAMod.f_function(xm_2,xm_1,xmk)
            ymk = np.append(ymk, ymk_next)
            ymk = ymk[2:]

            Ymk.append(ymk[-2:])
            Ypk.append(ypk)
            Upk.append(upk)
            print(dU_opt[:6])
            YspM.append(self.y_sp[0])
            YspP.append(self.y_sp[1])
            if i == 5:
                self.y_sp = np.array([[10], [8]])
                self.y_sp = ca.DM(self.iTil(self.y_sp,self.p).reshape(-1,1))
            
            print('teste')

        fig, axes = plt.subplots(2, 2, figsize=(12, 5), sharex=True)

        x = np.linspace(0,iter,iter)
        YspM = np.array(YspM)
        YspP = np.array(YspP)
        axes[0][0].plot(x, np.array(Ymk)[:,0], label="resM", color = 'green')
        axes[0][0].plot(x, np.array(Ypk)[:,0], label="plantaM", color="blue")
        axes[0][0].plot(x, YspM.squeeze(), linestyle="--", color="red", label="y_sp")
        axes[0][0].set_title("Vazão")
        axes[0][0].set_ylabel("Valor")
        axes[0][0].legend()
        axes[0][0].grid()
        axes[0][0].set_ylim(0,15)

        axes[0][1].plot(x, np.array(Ymk)[:,1], label="resP", color="green")
        axes[0][1].plot(x, np.array(Ypk)[:,1], label="plantaM", color="blue")
        axes[0][1].plot(x, YspP.squeeze(), linestyle="--", color="red", label="y_sp")
        axes[0][1].set_title("Pressão")
        axes[0][1].legend()
        axes[0][1].grid()
        axes[0][1].set_ylim(0,15)
        
        axes[1][0].plot(x, np.array(Upk)[:,0], label="resP", color="green")
        axes[1][0].set_title("Alpha")
        axes[1][1].plot(x, np.array(Upk)[:,1], label="resP", color="green")
        axes[1][1].set_title("Velocidade de rotacao")

        plt.xlabel("Tempo")
        plt.suptitle("Comparação de resM e resP")
        plt.tight_layout()
        

        plt.show()

        return dU_opt

if __name__ == '__main__':
    p, m, q, r, steps = 50, 3, [8/100,10/100], [1/0.15**2, 1/5000**2], 3
    mpc = PINN_MPC(p, m, q, r, steps)
    dU_opt = mpc.run()
    print("Controle ótimo:", dU_opt, dU_opt.shape)
