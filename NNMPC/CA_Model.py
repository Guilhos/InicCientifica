import numpy as np
import torch
import matplotlib.pyplot as plt
import casadi as ca


class CA_Model:
    def __init__(self, modelpath):
        # Carregar os pesos do modelo salvos
        model_path = modelpath
        state_dict = torch.load(model_path)
        
        Wi = state_dict['rnn_layer.weight_ih_l0'][:].numpy()
        Wh = state_dict['rnn_layer.weight_hh_l0'][:].numpy()
        Bi = state_dict['rnn_layer.bias_ih_l0'][:].numpy()
        Bh = state_dict['rnn_layer.bias_hh_l0'][:].numpy()
        print(Wi.shape, Wh.shape, Bi.shape, Bh.shape)
        
        Wii = ca.DM(Wi[:60])
        Wif = ca.DM(Wi[60:120])
        Wig = ca.DM(Wi[120:180])
        Wio = ca.DM(Wi[180:240])
        Wi = [Wii,Wif,Wig,Wio]
        
        Whi = ca.DM(Wh[:60])
        Whf = ca.DM(Wh[60:120])
        Whg = ca.DM(Wh[120:180])
        Who = ca.DM(Wh[180:240])
        Wh = [Whi, Whf, Whg, Who]
        
        Bii = ca.DM(Bi[:60].reshape(-1, 1))
        Bif = ca.DM(Bi[60:120].reshape(-1, 1))
        Big = ca.DM(Bi[120:180].reshape(-1, 1))
        Bio = ca.DM(Bi[180:240].reshape(-1, 1))
        Bi = [Bii,Bif,Big,Bio]
        
        Bhi = ca.DM(Bh[:60].reshape(-1, 1))
        Bhf = ca.DM(Bh[60:120].reshape(-1, 1))
        Bhg = ca.DM(Bh[120:180].reshape(-1, 1))
        Bho = ca.DM(Bh[180:240].reshape(-1, 1))
        Bh = [Bhi,Bhf,Bhg,Bho]
        
        LSTM = [Wi,Wh,Bi,Bh]
        
        Wd1 = ca.DM(state_dict['dense_layers.0.weight'].numpy())
        Bd1 = ca.DM(state_dict['dense_layers.0.bias'].numpy())
        Dense1 = [Wd1,Bd1]
        
        Wd2 = ca.DM(state_dict['dense_layers.2.weight'].numpy())
        Bd2 = ca.DM(state_dict['dense_layers.2.bias'].numpy())
        Dense2 = [Wd2,Bd2]
        
        #print(Wd1.shape, Bd1.shape, Wd2.shape, Bd2.shape)
        
        # Organizar os pesos e bias no formato esperado
        params = [LSTM,Dense1,Dense2]

        self.x_min = ca.DM(np.array([[3.4846e+00], [5.2707e+00], [3.4151e-01], [2.5049e+04]]))
        self.x_max = ca.DM(np.array([[1.2275e+01], [1.0323e+01], [6.5596e-01], [5.2308e+04]]))

        self.params = params
        self.f_function = self._build_model()

    def sigmoid(self, x):
        return 1 / (1 + ca.exp(-x))

    def lstm_step(self, x_t, h_t, c_t, Wi, Wh, bi, bh):
        """ Executa uma única etapa da célula LSTM. """
        
        # Gates
        i_t = self.sigmoid(Wi[0] @ x_t + Wh[0] @ h_t + bi[0] + bh[0])  # Forget gate
        f_t = self.sigmoid(Wi[1] @ x_t + Wh[1] @ h_t + bi[1] + bh[1])  # Input gate
        o_t = self.sigmoid(Wi[3] @ x_t + Wh[3] @ h_t + bi[3] + bh[3])  # Output gate
        
        # Candidato à célula
        g_t = ca.tanh(Wi[2] @ x_t + Wh[2] @ h_t + bi[2] + bh[2])
        
        # Atualização da célula e estado oculto
        c_t_new = f_t * c_t + i_t * g_t
        h_t_new = o_t * ca.tanh(c_t_new)
        
        return h_t_new, c_t_new

    def lstm_layer(self, x_seq, Wi, Wh, bi, bh, h0, c0):
        """ Executa uma camada LSTM para uma sequência de entrada. """
        h_t, c_t = h0, c0
        
        for x_t in x_seq:
            h_t, c_t = self.lstm_step(x_t, h_t, c_t, Wi, Wh, bi, bh)
        
        return ca.horzcat(h_t)  # Concatena ao longo do tempo

    def dense_layer(self, x, W, b, activation='tanh'):
        """ Camada densa com ativação """
        z = W @ x + b
        return ca.tanh(z) if activation == 'tanh' else z

    def _build_model(self):
        x_seq = [ca.MX.sym(f'x_{i}', 4, 1) for i in range(3)]  # Sequência de entrada simbólica

        x_seq_norm = [(2 * (x - self.x_min) / (self.x_max - self.x_min) - 1) for x in x_seq]

        params = self.params
        
        output = self.dense_layer(
            self.dense_layer(
                self.lstm_layer(x_seq_norm, params[0][0], params[0][1], params[0][2], params[0][3], np.zeros((60, 1)), np.zeros((60, 1))),
                params[1][0], params[1][1]
            ),
            params[2][0], params[2][1]
        )

        output = ((output + 1) / 2) * (self.x_max[:2] - self.x_min[:2]) + self.x_min[:2]
        
        return ca.Function('CA_Model', x_seq, [output])

    def build_model_mpc(self,p,m,ny,nu,steps):

        y_k = ca.MX.sym(f'y_{i}', ny*steps, 1) # Sequência de entrada simbólica

        u_k = ca.MX.sym(f'u_{i}', nu*steps, 1)  # Sequência de entrada simbólica

        du_k = ca.MX.sym(f'du_{i}', nu*m, 1) # Sequência de entrada simbólica

        for i in range(m):
            u_k[-2*(3-i):-2*(3-i)+1] = u_k[-2*(3-i):-2*(3-i)+1] + du_k[2*i:2*i+1]
        
        x_seq = [[y_k[-ny*steps],y_k[-ny*steps+1],u_k[-nu*steps],u_k[-nu*steps+1]],
                 [y_k[-ny*(steps-1)],y_k[-ny*(steps-1)+1],u_k[-nu*(steps-1)],u_k[-nu*(steps-1)+1]],
                 [y_k[-ny*(steps-2)],y_k[-ny*(steps-2)+1],u_k[-nu*(steps-2)],u_k[-nu*(steps-2)+1]]]

        x_seq_norm = [(2 * (x - self.x_min) / (self.x_max - self.x_min) - 1) for x in x_seq]

        params = self.params
        
        output = self.dense_layer(
            self.dense_layer(
                self.lstm_layer(x_seq_norm, params[0][0], params[0][1], params[0][2], params[0][3], np.zeros((60, 1)), np.zeros((60, 1))),
                params[1][0], params[1][1]
            ),
            params[2][0], params[2][1]
        )

        output = ((output + 1) / 2) * (self.x_max[:2] - self.x_min[:2]) + self.x_min[:2]

        y_k.append(output)

        return ca.Function('CA_Model_MPC', [y_k, u_k, du_k], [y_k])

        

if __name__ == '__main__':
    from libs.simulationn import Simulation

    p = 50
    m = 3

    sim = Simulation(3,3)
    sim_mf = Simulation(1,1)
    y0, u0 = sim.pIniciais()
    nU = len(u0) / m
    dU = [[0],[0],[0],[0],[0],[0]]
    dU = np.concatenate((np.array(dU), np.zeros((int(nU) * (p-m), 1))))
    yPlanta = sim_mf.pPlanta(y0,dU)
    #print(y0, u0)
    x0 = []

    # Iterar pelos dois arrays e intercalar 2 elementos de cada vez
    for i in range(0, len(y0), 2):
        x0.append(y0[i:i+2])  # Adiciona 2 elementos de y
        x0.append(u0[i:i+2])  # Adiciona 2 elementos de u

    # Transformar o x0ado em um array numpy
    x0 = np.vstack(x0)
    #print(x0)

    # Criar a instância da rede CasADi
    Modelo = CA_Model("NNMPC/libs/modelo_treinado.pth")
    
    # Extrai cada linha de x0
    #print(Modelo.f_function)

    for i in range(1):
        saida = Modelo.f_function(ca.vertcat(y0[-6:-4],u0[-6:-4]),ca.vertcat(y0[-4:-2],u0[-4:-2]),ca.vertcat(y0[-2:],u0[-2:]))
        x0 = np.append(x0,saida)
        x0 = np.append(x0,x0[-4:-2])

    print(x0.shape)
    print("Saída da rede CasADi:", saida)
    print(yPlanta)
    #print(yPlanta - saida)
    #print('AAAA')