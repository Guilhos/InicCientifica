import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import time
from libs.simulationn import Simulation
from libs.Interpolation import Interpolation

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
        interp = Interpolation('NNMPC/libs/tabela_phi.csv')
        interp.load_data()
        self.lut = interp.interpolate()

        # Limites das variáveis
        self.u_min = np.array([[0.35], [27e3]])
        self.u_max = np.array([[0.65], [5e4]])
        self.dU_min = np.array([[-0.1], [-2500]])
        self.dU_max = np.array([[0.1], [2500]])
        self.y_min = np.array([[3.5], [5.27]])
        self.y_max = np.array([[12.3], [9.3]])

        self.params = [-25.0181, 42.0452, -17.9068, 3.0313]

        # Setpoint provisório
        self.y_sp = np.array([[8.5], [6.9]])
    
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
    
    def f_vazaoMin(self):
        x = ca.MX.sym('x', 1)
        f = self.params[0] + self.params[1]*x + self.params[2]*x**2 + self.params[3]*x**3
        return ca.Function('f_vazaoMin', [x], [f])

    def ajusteMatrizes(self):
        self.y_sp = ca.DM(self.iTil(self.y_sp,self.p).reshape(-1,1)) # Expansão do y_setpoint para P. SHAPE -> (nY*P, 1) # Expansão do y_min para P. SHAPE -> (nY*P, 1)
        self.y_max = ca.DM(self.iTil(self.y_max,self.p)) # Expansão do y_max para P. SHAPE -> (nY*P, 1)

        self.u_min = ca.DM(self.iTil(self.u_min,self.m)) # Expansão do u_min para M. SHAPE -> (nU*M, 1)
        self.u_max = ca.DM(self.iTil(self.u_max,self.m)) # Expansão do u_max para M. SHAPE -> (nU*M, 1)

        self.dU_min = self.iTil(self.dU_min, self.m) # Expansão do dU_min para M. SHAPE -> (nU*M, 1)
        #self.dU_min = ca.DM(np.concatenate((self.dU_min,np.zeros((int(self.nU) * (self.p - self.m), 1))))) # Adição de P - M linhas de 0. SHAPE -> (nU*P, 1)
        self.dU_max = self.iTil(self.dU_max, self.m) # Expansão do dU_max para M. SHAPE -> (nU*M, 1)
        #self.dU_max = ca.DM(np.concatenate((self.dU_max,np.zeros((int(self.nU) * (self.p - self.m), 1))))) # Adição de P - M linhas de 0. SHAPE -> (nU*P, 1)

        self.q = ca.DM(np.diag(np.array([self.q[0],self.q[1]] * (self.nY*self.p // 2)))) # Criação de uma matriz com os valores de Q na diagonal. SHAPE -> (nY*p, nY*p)
        self.r = ca.DM(np.diag(np.array([self.r[0],self.r[1]] * (self.nU*self.m // 2)))) # Criação de uma matriz com os valores de R na diagonal. SHAPE -> (nU*p, nU*p)

    def nlp_func(self):
        # Criando o problema de otimização
        opti = ca.Opti()

        # Definição das variáveis de decisão
        dUs = opti.variable(self.nU * self.m, 1)
        Fs  = opti.variable(1, 1)  # Variável escalar para Fs
        yModelk = opti.parameter(self.nY * self.steps, 1)
        uModelk = opti.parameter(self.nU * self.steps, 1)
        yPlantak = opti.parameter(self.nY, 1)
        ysp = opti.parameter(self.nY * self.p, 1)# yPlantak como parâmetro
        alphak = opti.parameter(3, 1)  # Parâmetro para alphas
        nrotk = opti.parameter(3, 1)  # Parâmetro para N_RotS
        y_min = opti.parameter(self.nY * self.p, 1)
    
        x = ca.vertcat(dUs, Fs)
        
        # Definição do problema de otimização
        opti.minimize(Fs)

        # Erro entre a planta e o modelo
        dYk = yPlantak - yModelk[-self.nY:]
        dYk = ca.repmat(dYk, self.p, 1)

        # Predição do modelo
        yModel_pred, _,_ = self.sim_pred.ca_YPredFun(yModelk, dUs, alphak, nrotk)
        
        # Matriz triangular para os controles
        matriz_inferior = self.matriz_triangular_identidade(self.m, self.m, self.nU)

        # Restrições
        # x_min e x_max
        opti.subject_to(opti.bounded(self.dU_min, dUs, self.dU_max))
        opti.subject_to(opti.bounded(0, Fs, 10e23))

        # lbg e ubg
        opti.subject_to(opti.bounded(y_min, yModel_pred + dYk, self.y_max))
        opti.subject_to(opti.bounded(self.u_min, ca.repmat(uModelk[-2:], self.m, 1) + matriz_inferior @ dUs, self.u_max))
        opti.subject_to(Fs - ((yModel_pred - ysp + dYk).T @ self.q @ (yModel_pred - ysp + dYk) + dUs.T @ self.r @ dUs) == 0)  # Restrições de igualdade

        opti.solver('ipopt', {
            "ipopt.print_level": 0,
            "ipopt.tol": 1e-6,                      # Tolerância do solver (pode ajustar entre 1e-4 e 1e-8)
            "ipopt.constr_viol_tol": 1e-8,          
            "ipopt.max_iter": 750,                   # Reduz número de iterações (ajustável)
            "ipopt.mu_strategy": "adaptive",         # Estratégia de barreira mais eficiente
            "ipopt.linear_solver": "mumps",          # Solver linear mais rápido para problemas médios/grandes
            "ipopt.sb": "yes"
        })
        print(opti)

        # Criando a função otimizada
        return opti.to_function(
            "opti_nlp",
            [yModelk, uModelk, yPlantak, ysp, dUs, Fs, alphak, nrotk, y_min],
            [x]
        )

    def otimizar(self, ymk, umk, ypk, alphaK, nrotK):
        dYk = ypk - ymk[-self.nY:]
        dYk = ca.repmat(dYk, self.p, 1)
        dU_init = self.dU
        yModel_init, _, _ = self.sim_pred.ca_YPredFun(ymk,dU_init,alphaK, nrotK)
        yModel_init = np.array(yModel_init.full())
        Fs_init = (yModel_init - self.y_sp + dYk).T @ self.q @ (yModel_init - self.y_sp + dYk) + dU_init.T @ self.r @ dU_init
        y_minAux = ca.DM(self.iTil(self.y_min,self.p))

        x_opt = self.opti_nlp(ymk, umk, ypk, self.y_sp, dU_init, Fs_init, alphaK, nrotK, y_minAux)

        dU_opt = x_opt[:self.nU * self.m]
        dU_opt = np.array(dU_opt.full())
        return dU_opt
    
    def run(self):
        self.ajusteMatrizes()
        ymk = np.tile(np.array([[8.5], [6.9]]), (self.steps, 1))
        umk = np.tile(np.array([[0.5], [385e2]]), (self.steps, 1))
        ypk = ymk[-self.nY:]
        ymk_next = ypk
        self.opti_nlp = self.nlp_func()
        alphaK = self.sim_mf.alphas
        nrotK = self.sim_mf.N_RotS
        self.mMin = self.f_vazaoMin()

        Ypk = []
        Upk = []
        dURot = []
        dUAlpha = []
        Ymk = []
        YspM = []
        YspP = []
        YmMin = []
        Tempos = []
        phi = []
        #Ymink = []
        #Ymaxk = []

        iter = 130
        for i in range(iter): 
            x = self.lut(ca.vertcat(umk[-1],ypk[-2]))
            mMink = self.mMin(x)
            mMink = np.array(mMink.full())
            self.y_min = np.array([[float(mMink[0][0])], [5.27]])

            t1 = time.time()
            print(15*'='+ f'Iteração {i+1}' + 15*'=')
            dU_opt = self.otimizar(ymk, umk, ypk, alphaK, nrotK)
            
            self.dUk = dU_opt[:self.nU]
            self.dU = dU_opt
            
            umk = umk.reshape(6, 1)
            umk = np.append(umk, umk[-self.nU:] + self.dUk)
            umk = umk[self.nU:]

            ymk_next, alphaK, nrotK = self.sim_mf.ca_YPredFun(ymk, dU_opt, alphaK, nrotK)
            ymk_next = np.array(ymk_next.full())
            
            t2 =  time.time()
            Tempos.append(t2-t1)
            print(f'Tempo decorrido: {t2-t1}')
            
            ypk, upk = self.sim_mf.pPlanta(ypk, self.dUk)

            print('dYk: ',ymk_next - ypk)
            
            upk = upk.flatten()
            ypk = ypk.flatten()
            
            ymk = np.append(ymk, ymk_next)
            ymk = ymk[self.nY:]

            Ymk.append(ymk_next)
            Ypk.append(ypk)
            Upk.append(upk)
            dUAlpha.append(self.dUk[0])
            dURot.append(self.dUk[1])
            print('dUk: ',dU_opt[:self.m*self.nU])
            YspM.append(self.y_sp[0])
            YspP.append(self.y_sp[1])
            YmMin.append(self.y_min[0])
            phi.append(x)
            
            if i == 10:
                self.y_sp = np.array([[7.1], [6.2]])
                self.y_sp = ca.DM(self.iTil(self.y_sp,self.p).reshape(-1,1))
            elif i == 50:
                self.y_sp = np.array([[9], [7.05]])
                self.y_sp = ca.DM(self.iTil(self.y_sp,self.p).reshape(-1,1))
            elif i == 90:
                self.y_sp = np.array([[8], [6.6]])
                self.y_sp = ca.DM(self.iTil(self.y_sp,self.p).reshape(-1,1))
                
        #self.plot_results(iter, Ymk, Ypk, Upk, YspM, YspP, Tempos)
        
        return iter, Ymk, Ypk, Upk, dURot, dUAlpha, YspM, YspP, YmMin, Tempos, phi
            
    def plot_results(self, iter, Ymk, Ypk, Upk, YspM, YspP, Tempos):

        fig, axes = plt.subplots(3, 2, figsize=(12, 8))

        x = np.linspace(0, iter, iter)
        YspM = np.array(YspM)
        YspP = np.array(YspP)

        # Vazão x Tempo
        axes[0][0].plot(x / 2, np.array(Ymk)[:, 0], label="Modelo", color='green')
        axes[0][0].plot(x / 2, np.array(Ypk)[:, 0], label="Planta", color="blue")
        axes[0][0].plot(x / 2, YspM.squeeze(), linestyle="--", color="red", label="y_sp")
        axes[0][0].plot([0, iter / 2], [3.5, 3.5], linestyle="--", color="black")
        axes[0][0].plot([0, iter / 2], [12.3, 12.3], linestyle="--", color="black")
        axes[0][0].set_title("Vazão x Tempo")
        axes[0][0].set_ylabel("Vazão / kg/s")
        axes[0][0].set_xlabel("Tempo / s")
        axes[0][0].legend()
        axes[0][0].grid()
        axes[0][0].set_ylim(3, 12.8)

        # Pressão x Tempo
        axes[0][1].plot(x / 2, np.array(Ymk)[:, 1], label="Modelo", color="green")
        axes[0][1].plot(x / 2, np.array(Ypk)[:, 1], label="Planta", color="blue")
        axes[0][1].plot(x / 2, YspP.squeeze(), linestyle="--", color="red", label="y_sp")
        axes[0][1].plot([0, iter / 2], [5.27, 5.27], linestyle="--", color="black")
        axes[0][1].plot([0, iter / 2], [9.3, 9.3], linestyle="--", color="black")
        axes[0][1].set_title("Pressão x Tempo")
        axes[0][1].set_ylabel("Pressão / kPa")
        axes[0][1].set_xlabel("Tempo / s")
        axes[0][1].legend()
        axes[0][1].grid()
        axes[0][1].set_ylim(4.77, 9.83)

        # Abertura da Válvula x Tempo
        axes[1][0].plot(x / 2, np.array(Upk)[:, 0], label="Alpha", color="green")
        axes[1][0].plot([0, iter / 2], [0.35, 0.35], linestyle="--", color="black")
        axes[1][0].plot([0, iter / 2], [0.65, 0.65], linestyle="--", color="black")
        axes[1][0].set_title("Abertura da Válvula x Tempo")
        axes[1][0].set_ylabel("Alpha / %")
        axes[1][0].set_xlabel("Tempo / s")
        axes[1][0].legend()
        axes[1][0].grid()

        # Velocidade de Rotação x Tempo
        axes[1][1].plot(x / 2, np.array(Upk)[:, 1], label="N", color="green")
        axes[1][1].plot([0, iter / 2], [27e3, 27e3], linestyle="--", color="black")
        axes[1][1].plot([0, iter / 2], [5e4, 5e4], linestyle="--", color="black")
        axes[1][1].set_title("Velocidade de Rotação x Tempo")
        axes[1][1].set_ylabel("N / Hz")
        axes[1][1].set_xlabel("Tempo / s")
        axes[1][1].legend()
        axes[1][1].grid()

        # Tempo por Iteração
        indice_max = Tempos.index(max(Tempos))
        for i, tempo in enumerate(Tempos):
            cor = "red" if i == indice_max else "blue"
            axes[2][0].bar(x[i], tempo, color=cor)
        axes[2][0].plot([0, iter], [0.5, 0.5], linestyle="--", color="black")
        axes[2][0].plot([0, iter], [np.mean(Tempos), np.mean(Tempos)], linestyle="--", color="red", label=f"Média: {np.mean(Tempos):.2f} s")
        axes[2][0].set_title("Tempo por Iteração")
        axes[2][0].set_ylabel("Tempo / s")
        axes[2][0].set_xlabel("Iteração")
        axes[2][0].legend()
        axes[2][0].grid()

        # Histograma das Frequências de Tempo
        axes[2][1].hist(Tempos, bins=20, color='blue', alpha=0.7, edgecolor='black')
        axes[2][1].set_title("Histograma das Frequências de Tempo")
        axes[2][1].set_xlabel("Tempo")
        axes[2][1].set_ylabel("Frequência")

        plt.suptitle("Resultados NMPC - CasADi", fontsize=16)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':

    qVazao = 1/12.5653085708618164062**2
    qPressao = 0.1/9.30146217346191406250**2
    rAlpha = 0/0.15**2
    rN = 1e-4/5000**2

    p, m, q, r, steps = 12, 3, [qVazao,qPressao], [rAlpha, rN], 3
    mpc = Only_NMPC(p, m, q, r, steps)
    dU_opt = mpc.run()
