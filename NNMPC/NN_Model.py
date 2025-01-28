import numpy as np
import torch
import onnxruntime as ort
import matplotlib.pyplot as plt
import casadi as ca

class NN_Model:
    def __init__(self, p=50, m = 3):
        self.p = p # Horizonte de Predição
        self.m = m # Horizonte de Controle

        self.onnx_model_path = "NNMPC/libs/modelo.onnx"
        self.onnx_session = ort.InferenceSession(self.onnx_model_path)
        self.input_tensor = torch.zeros((1, 3, 4), dtype=torch.float32)

    def run(self,y0,u0,dU):
        vazaoMassica = [y0[0, 0].item(), y0[2, 0].item(), y0[4, 0].item()]
        pressaoPlenum = [y0[1, 0].item(), y0[3, 0].item(), y0[5, 0].item()]
        alpha = [u0[0, 0].item(), u0[2, 0].item(), u0[4, 0].item()]
        N_Rot = [u0[1, 0].item(), u0[3, 0].item(), u0[5, 0].item()]

        for k in range(self.p):
            # Atualizar os valores do tensor diretamente
            self.input_tensor[0, :, 0] = torch.tensor(vazaoMassica[-3:])
            self.input_tensor[0, :, 1] = torch.tensor(pressaoPlenum[-3:])
            self.input_tensor[0, :, 2] = torch.tensor(alpha[-3:])
            self.input_tensor[0, :, 3] = torch.tensor(N_Rot[-3:])
            # Previsão com desativação do gradiente
            onnx_inputs = {'input': self.input_tensor.numpy()}
            onnx_outputs = self.onnx_session.run(None, onnx_inputs)
            dUk = dU[2*k:2*(k+1)]
            
            # Adicionar previsões diretamente
            if k < m:
                alpha.append(alpha[-1] + dUk[0].item())
                N_Rot.append(N_Rot[-1] + dUk[1].item())
            vazaoMassica.append(onnx_outputs[0][0, 0, 0])
            pressaoPlenum.append(onnx_outputs[0][0, 0, 1])

        y = (vazaoMassica[3:], pressaoPlenum[3:])
        y = np.array(y).T.flatten().reshape(-1, 1)
        u = (alpha[3:], N_Rot[3:])
        u = np.array(u).T.flatten().reshape(-1, 1)

        return y,u
            
import casadi as ca

class RedeCasADiRNN:
    def __init__(self, pesos_e_bias):
        self.pesos_e_bias = pesos_e_bias
        self.funcao = self._criar_rede_casadi()

    def _criar_rede_casadi(self):
        # Variáveis de entrada: duas matrizes separadas
        x = ca.MX.sym('x', 1,3,4)  # Vetor com variáveis controladas (massa e pressão)

        # Inicializar estados ocultos e células da RNN
        hidden_size = self.pesos_e_bias['rnn_layer.weight_hh_l0'].shape[1]  # Deverá ser 60
        h_t = ca.MX.zeros((1, hidden_size))  # Estado oculto com dimensão 60

        # Pesos da RNN
        W_ih = self.pesos_e_bias['rnn_layer.weight_ih_l0']  # Shape (240, 6)
        W_hh = self.pesos_e_bias['rnn_layer.weight_hh_l0']  # Shape (240, 60)
        b_ih = self.pesos_e_bias['rnn_layer.bias_ih_l0']
        b_hh = self.pesos_e_bias['rnn_layer.bias_hh_l0']

        # Loop sobre os passos de tempo (4 elementos combinados da entrada)
        for t in range(4):
            # Cada x_t será uma fatia de 6 elementos do vetor combinado x_combined
            x_t = x_combined[3 * t : 3 * (t + 1)]  # Pegando as entradas do tempo t

            # Certifique-se de que x_t tem o formato correto (6, 1)
            x_t = ca.vertcat(x_t[:3], x_t[3:])  # Garantindo que x_t seja uma matriz de (6, 1)

            # Calcular a saída da RNN
            rnn_output = ca.mtimes(W_ih, x_t) + ca.mtimes(W_hh, h_t.T) + (b_ih + b_hh)

            # Atualizar o estado oculto
            h_t = ca.tanh(rnn_output)  # Tanh é usada como a função de ativação da RNN

        # Achatar a saída da RNN para alimentar na camada densa
        h_flat = h_t.T  # Saída já está no formato (1, hidden_size)

        # Camada densa 1 com tanh
        W1 = self.pesos_e_bias['dense_layers.0.weight']
        b1 = self.pesos_e_bias['dense_layers.0.bias']
        h1 = ca.tanh(ca.mtimes(W1, h_flat) + b1)

        # Camada densa 2 com tanh
        W2 = self.pesos_e_bias['dense_layers.2.weight']
        b2 = self.pesos_e_bias['dense_layers.2.bias']
        h2 = ca.tanh(ca.mtimes(W2, h1) + b2)

        # Criar a função CasADi
        return ca.Function('rede', [y, u], [h2])

    def avaliar(self, entrada1, entrada2):
        """
        Avalia a rede para duas entradas separadas.
        
        :param entrada1: Array (6, 1) para Massa e Pressão.
        :param entrada2: Array (6, 1) para Alpha e N.
        :return: Saída da rede neural.
        """
        return self.funcao(entrada1, entrada2)



if __name__ == '__main__':
    from libs.simulationn import Simulation

    p = 50
    m = 3

    sim = Simulation()
    y0, u0 = sim.run()
    nU = len(u0) / m
    dU = [[0.05],[2000],[-0.02],[-1000],[0.1],[1500]]
    dU = np.concatenate((np.array(dU), np.zeros((int(nU) * (p-m), 1))))
    print(y0.shape, u0.shape)

    # Carregar os pesos do modelo salvos
    model_path = "NNMPC/libs/modelo_treinado.pth"
    state_dict = torch.load(model_path)
    state_dict = torch.load(model_path)
    print(state_dict.keys())

    # Extrair os pesos e vieses como numpy arrays
    pesos_e_bias = {key: param.numpy() for key, param in state_dict.items()}

    # Criar a instância da rede CasADi
    rede_casadi = RedeCasADiRNN(pesos_e_bias)

    saida = rede_casadi.avaliar(y0, u0)

    print("Saída da rede CasADi:", saida)

    NNModel = NN_Model(p,m)
    y,u = NNModel.run(y0,u0,dU)
    print(y.shape,u.shape)

    '''
    pltz = np.linspace(1,p, p)
    fig, (ax1,ax2) = plt.subplots(2,1)
    ax1.plot(pltz, y[0], color='r')
    ax2.plot(pltz, y[1])
    plt.show()
    '''
