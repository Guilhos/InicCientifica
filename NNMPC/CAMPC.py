import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import time
from libs.simulationn import Simulation

class Only_NMPC():
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
        self.sim_pred = Simulation(p,m,steps)
        self.sim_mf = Simulation(1,1,steps)

        # Limites das variáveis
        self.u_min = np.array([[0.35], [27e3]])
        self.u_max = np.array([[0.65], [5e4]])
        self.dU_min = np.array([[-0.1], [-2500]])
        self.dU_max = np.array([[0.1], [2500]])
        self.y_min = np.array([[3.5], [5.27]])
        self.y_max = np.array([[12.3], [9.3]])

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

        self.u_min = ca.DM(self.iTil(self.u_min,self.m)) # Expansão do u_min para M. SHAPE -> (nU*M, 1)
        self.u_max = ca.DM(self.iTil(self.u_max,self.m)) # Expansão do u_max para M. SHAPE -> (nU*M, 1)

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
        dUs = opti.variable(self.nU * self.m, 1)
        Fs = opti.variable(1, 1)  # Variável escalar para Fs
        yModelk = opti.parameter(self.nY * self.steps, 1)
        uModelk = opti.parameter(self.nU * self.steps, 1)
        yPlantak = opti.parameter(self.nY, 1)
        ysp = opti.parameter(self.nY * self.p, 1)# yPlantak como parâmetro
    
        x = ca.vertcat(dUs, Fs)
        
        # Definição do problema de otimização
        opti.minimize(Fs)

        # Erro entre a planta e o modelo
        dYk = yPlantak - yModelk[-self.nY:]
        dYk = ca.repmat(dYk, self.p, 1)

        # Predição do modelo
        yModel_pred = self.sim_pred.pPlanta(yModelk, dUs, caller=self)
        
        # Matriz triangular para os controles
        matriz_inferior = self.matriz_triangular_identidade(self.m, self.m, self.nU)

        # Restrições
        # x_min e x_max
        opti.subject_to(opti.bounded(self.dU_min, dUs, self.dU_max))
        opti.subject_to(opti.bounded(0, Fs, 10e23))

        # lbg e ubg
        opti.subject_to(opti.bounded(self.y_min, yModel_pred, self.y_max))
        opti.subject_to(opti.bounded(self.u_min, ca.repmat(uModelk[-2:], self.m, 1) + matriz_inferior @ dUs, self.u_max))
        opti.subject_to(Fs - (yModel_pred - ysp + dYk).T @ self.q @ (yModel_pred - ysp + dYk) + dUs.T @ self.r @ dUs == 0)  # Restrições de igualdade

        opti.solver('ipopt', {
            "ipopt.print_level": 0,
            "ipopt.tol": 1e-4,                      # Tolerância do solver (pode ajustar entre 1e-4 e 1e-8)
            "ipopt.max_iter": 500,                   # Reduz número de iterações (ajustável)
            "ipopt.mu_strategy": "adaptive",         # Estratégia de barreira mais eficiente
            "ipopt.linear_solver": "mumps",          # Solver linear mais rápido para problemas médios/grandes
            "ipopt.sb": "yes"
        })
        print(opti)

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
        Fs_init = (ypk - self.y_sp + dYk).T @ self.q @ (ypk - self.y_sp + dYk) + dU_init.T @ self.r @ dU_init

        x_opt = self.opti_nlp(ymk, umk, ypk, self.y_sp, dU_init, Fs_init)

        dU_opt = x_opt[:self.nU * self.m]
        #stats = self.opti_nlp.stats()
        #print(stats)
        return dU_opt
    
    def run(self):
        self.ajusteMatrizes()
        xmk = []
        ymk, umk = self.sim_pred.pIniciais() # Recebe os pontos iniciais, ymk [6,1] umk [6,1]
        ypk = ymk[-self.nY:]
        ymk_next = ypk
        self.opti_nlp = self.nlp_func()

        Ypk = []
        Upk = []
        Ymk = []
        YspM = []
        YspP = []
        Tempos = []
        #Ymink = []
        #Ymaxk = []

        iter = 300
        for i in range(iter):
            t1 = time.time()
            print(15*'='+ f'Iteração {i+1}' + 15*'=')
            dU_opt= self.otimizar(ymk, umk, ypk)
            #print(stats)
            self.dUk = dU_opt[:self.nU]
            self.dU = ca.vertcat(dU_opt, np.zeros((self.nU, 1)))
            self.dU = self.dU[self.nU:]
            
            umk = np.append(umk, umk[-self.nU:] + self.dUk)
            umk = umk[self.nU:]

            ymk, _ = self.sim_mf.pPlanta(ymk, self.dUk)
            
            t2 =  time.time()
            Tempos.append(t2-t1)
            print(f'Tempo decorrido: {t2-t1}')
            
            ypk, upk = self.sim_mf.pPlanta(ypk, self.dUk)
            
            upk = upk.flatten()
            ypk = ypk.flatten()
            
            ymk = np.append(ymk, ymk_next)
            ymk = ymk[self.nY:]

            Ymk.append(ymk[-self.nY:])
            Ypk.append(ypk)
            Upk.append(upk)
            print(dU_opt[:self.m*self.nU])
            YspM.append(self.y_sp[0])
            YspP.append(self.y_sp[1])
            if i == 10:
                self.y_sp = np.array([[10.09972032], [6.89841795]])
                self.y_sp = ca.DM(self.iTil(self.y_sp,self.p).reshape(-1,1))
            elif i == 100:
                self.y_sp = np.array([[8.39637471], [6.4025308]])
                self.y_sp = ca.DM(self.iTil(self.y_sp,self.p).reshape(-1,1))
            elif i == 200:
                self.y_sp = np.array([[5.67905178], [5.85870524]])
                self.y_sp = ca.DM(self.iTil(self.y_sp,self.p).reshape(-1,1))
            

        fig, axes = plt.subplots(3, 2, figsize=(12, 8))

        x = np.linspace(0,iter,iter)
        YspM = np.array(YspM)
        YspP = np.array(YspP)
        axes[0][0].plot(x/2, np.array(Ymk)[:,0], label="resM", color = 'green')
        axes[0][0].plot(x/2, np.array(Ypk)[:,0], label="plantaM", color="blue")
        axes[0][0].plot(x/2, YspM.squeeze(), linestyle="--", color="red", label="y_sp")
        axes[0][0].plot([0,iter/2], [3.5,3.5], linestyle="--", color="black")
        axes[0][0].plot([0,iter/2], [12.3,12.3], linestyle="--", color="black")
        axes[0][0].set_title("Vazão x Tempo")
        axes[0][0].set_ylabel("Vazão / kg/s")
        axes[0][0].set_xlabel("Tempo / s")
        axes[0][0].legend()
        axes[0][0].grid()
        axes[0][0].set_ylim(3,12.8)

        axes[0][1].plot(x/2, np.array(Ymk)[:,1], label="resP", color="green")
        axes[0][1].plot(x/2, np.array(Ypk)[:,1], label="plantaM", color="blue")
        axes[0][1].plot(x/2, YspP.squeeze(), linestyle="--", color="red", label="y_sp")
        axes[0][1].plot([0,iter/2], [5.27,5.27], linestyle="--", color="black")
        axes[0][1].plot([0,iter/2], [9.3,9.3], linestyle="--", color="black")
        axes[0][1].set_title("Pressão x Tempo")
        axes[0][1].set_ylabel("Pressão / kPa")
        axes[0][1].set_xlabel("Tempo / s")
        axes[0][1].legend()
        axes[0][1].grid()
        axes[0][1].set_ylim(4.77,9.83)
        
        axes[1][0].plot(x/2, np.array(Upk)[:,0], label="Alpha", color="green")
        axes[1][0].plot([0,iter/2], [0.35,0.35], linestyle="--", color="black")
        axes[1][0].plot([0,iter/2], [0.65,0.65], linestyle="--", color="black")
        axes[1][0].set_title("Abertura da Válvula x Tempo")
        axes[1][0].set_ylabel("Alpha / %")
        axes[1][0].set_xlabel("Tempo / s")
        axes[1][0].legend()
        axes[1][0].grid()

        axes[1][1].plot(x/2, np.array(Upk)[:,1], label="N", color="green")
        axes[1][1].plot([0,iter/2], [27e3,27e3], linestyle="--", color="black")
        axes[1][1].plot([0,iter/2], [5e4,5e4], linestyle="--", color="black")
        axes[1][1].set_title("Velocidade de rotacao x Tempo")
        axes[1][1].set_ylabel("N / hz")
        axes[1][1].set_xlabel("Tempo / s")
        axes[1][1].legend()
        axes[1][1].grid()

        axes[2][0].plot(x, Tempos, label="Tempo", color="green")
        axes[2][0].plot([0,iter], [0.5,0.5],linestyle = "--", color="black")
        axes[2][0].set_title("Tempo por Iteração")
        axes[2][0].set_ylabel("Tempo / s")
        axes[2][0].set_xlabel("Iteração")
        axes[2][0].legend()
        axes[2][0].grid()
        
        axes[2][1].hist(Tempos, bins=20, color='blue', alpha=0.7, edgecolor='black')
        axes[2][1].set_title("Histograma das Frequências de Tempo")
        axes[2][1].set_xlabel("Tempo")
        axes[2][1].set_ylabel("Frequência")

        plt.suptitle("Comparação de resM e resP")
        plt.tight_layout()

        plt.show()

        return dU_opt

if __name__ == '__main__':

    qVazao = 1/12.5653085708618164062**2
    qPressao = 0.1/9.30146217346191406250**2
    rAlpha = 0/0.15**2
    rN = 1e-4/5000**2


    p, m, q, r, steps = 12, 3, [qVazao,qPressao], [rAlpha, rN], 3
    mpc = Only_NMPC(p, m, q, r, steps)
    dU_opt = mpc.run()
    print("Controle ótimo:", dU_opt, dU_opt.shape)
