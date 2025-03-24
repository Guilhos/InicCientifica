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
        # Criando o problema de otimização
        opti = ca.Opti()

        # Definição das variáveis de decisão
        yModelk = opti.parameter(self.nY * self.steps, 1)
        uModelk = opti.parameter(self.nU * self.steps, 1)
        yPlantak = opti.parameter(self.nY, 1)
        ysp = opti.parameter(self.nY * self.p, 1)# yPlantak como parâmetro
        dUs = opti.variable(self.nU * self.m, 1)
        Fs = opti.variable(1, 1)  # Variável escalar para Fs

        x = ca.vertcat(dUs, Fs)

        # Erro entre a planta e o modelo
        dYk = yPlantak - yModelk[-self.nY:]
        dYk = ca.repmat(dYk, self.p, 1)

        # Conversão de constantes para CasADi MX
        dU_init = ca.DM(self.dU)
        rF = ca.MX(self.r)
        qF = ca.MX(self.q)

        # Predição do modelo
        yModel_pred = self.CAMod.pred_function(yModelk, uModelk, dUs)
        
        # Matriz triangular para os controles
        matriz_inferior = self.matriz_triangular_identidade(self.steps, self.steps, self.nU)

        # Função objetivo (custo)
        custo = (yModel_pred - ysp + dYk).T @ qF @ (yModel_pred - ysp + dYk) + dU_init.T @ rF @ dU_init

        # Restrições
        # x_min e x_max
        opti.subject_to(opti.bounded(self.dU_min, dUs, self.dU_max))
        opti.subject_to(opti.bounded(0, Fs, 10e23))

        # lbg e ubg
        opti.subject_to(opti.bounded(self.y_min, yModel_pred, self.y_max))
        opti.subject_to(opti.bounded(self.u_min, ca.repmat(uModelk[-2:], self.steps, 1) + matriz_inferior @ dUs, self.u_max))
        opti.subject_to(Fs - custo == 0)  # Restrições de igualdade

        # Definição do problema de otimização
        opti.minimize(Fs)
        opti.solver('ipopt', {
            "print_level": 0,            # Desativa a saída do IPOPT
            "print_timing_statistics": "no",  # Remove estatísticas de tempo
            "sb": "yes"                  # "Silent barrier" - silencia IPOPT
        })

        # Criando a função otimizada
        return opti.to_function(
            "opti_nlp",
            [yModelk, uModelk, yPlantak, ysp, dUs, Fs],
            [x]
        )

    def otimizar(self, ymk, umk, ypk):
        dYk = ypk - ymk[-self.nY:]
        dYk = ca.repmat(dYk, self.p, 1)
        dU_init = self.dU
        yModel_init = self.CAMod.pred_function(ymk,umk,dU_init)
        Fs_init = (yModel_init - self.y_sp + dYk).T @ self.q @ (yModel_init - self.y_sp + dYk) + dU_init.T @ self.r @ dU_init

        x_opt = self.opti_nlp(ymk, umk, ypk, self.y_sp, dU_init, Fs_init)
        dU_opt = x_opt[:self.nU * self.m]
        #stats = self.opti_nlp.stats()
        #print(stats)
        return dU_opt
    
    def run(self):
        self.ajusteMatrizes()
        xmk = []
        ymk, umk = self.sim_pred.pIniciais() # Recebe os pontos iniciais, ymk [6,1] umk [6,1]
        ypk, uk = ymk[-self.nY:], umk[-self.nU:]

        self.opti_nlp = self.nlp_func()

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
            dU_opt= self.otimizar(ymk, umk, ypk)
            #print(stats)
            self.dUk = dU_opt[:self.nU]
            self.dU = ca.vertcat(dU_opt, np.zeros((self.nU, 1)))
            self.dU = self.dU[2:]
            # if stats == 'Solve_Succeeded':
            #     self.dUk = dU_opt[:self.nU]
            #     self.dU = ca.vertcat(dU_opt, np.zeros((self.nU, 1)))
            #     self.dU = self.dU[2:]
            # elif stats == 'Infeasible_Problem_Detected':
            #     self.dUk = self.dU[:self.nU]
            #     self.dU = ca.vertcat(self.dU, np.zeros((self.nU, 1)))
            #     self.dU = self.dU[2:]
            
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
    p, m, q, r, steps = 50, 3, [8/100,1/100], [1/0.15**2, 1000/5000**2], 3
    mpc = PINN_MPC(p, m, q, r, steps)
    dU_opt = mpc.run()
    print("Controle ótimo:", dU_opt, dU_opt.shape)
