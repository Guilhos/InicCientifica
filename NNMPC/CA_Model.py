import numpy as np
import torch
import matplotlib.pyplot as plt
import casadi as ca


class CA_Model:
    def __init__(self, params):
        self.params = params
        self.lstm_function = self._build_model()

    def sigmoid(x):
        return 1 / (1 + ca.exp(-x))

    def lstm_step(self, x_t, h_t, c_t, W, b):
        """ Executa uma única etapa da célula LSTM. """
        concat_input = ca.vertcat(x_t, h_t)
        
        # Gates
        f_t = self.sigmoid(W['ih'] @ concat_input + W['hh'] @ h_t + b['ih'] + b['hh'])  # Forget gate
        i_t = self.sigmoid(W['ih'] @ concat_input + W['hh'] @ h_t + b['ih'] + b['hh'])  # Input gate
        o_t = self.sigmoid(W['ih'] @ concat_input + W['hh'] @ h_t + b['ih'] + b['hh'])  # Output gate
        
        # Candidato à célula
        c_hat_t = ca.tanh(W['ih'] @ concat_input + W['hh'] @ h_t + b['ih'] + b['hh'])
        
        # Atualização da célula e estado oculto
        c_t_new = f_t * c_t + i_t * c_hat_t
        h_t_new = o_t * ca.tanh(c_t_new)
        
        return h_t_new, c_t_new

    def lstm_layer(self, x_seq, W, b, h0, c0):
        """ Executa uma camada LSTM para uma sequência de entrada. """
        h_t, c_t = h0, c0
        h_seq = []
        
        for x_t in x_seq:
            h_t, c_t = self.lstm_step(x_t, h_t, c_t, W, b)
            h_seq.append(h_t)
        
        return ca.horzcat(*h_seq)  # Concatena ao longo do tempo

    def dense_layer(self, x, W, b, activation='tanh'):
        """ Camada densa com ativação """
        z = W @ x + b
        return ca.tanh(z) if activation == 'tanh' else z

    def _build_model(self):
        x_seq = [ca.MX.sym(f'x_{i}', 4, 1) for i in range(3)]  # Sequência de entrada simbólica

        params = self.params
        
        output = self.dense_layer(
            self.dense_layer(
                self.lstm_layer(x_seq, params['W_lstm'], params['b_lstm'], np.random.rand(60, 1), np.random.rand(60, 1)),
                params['W_dense1'], params['b_dense1']
            ),
            params['W_dense2'], params['b_dense2']
        )
        
        return ca.Function('LSTM_Model', x_seq + [ca.MX.sym('h0', 60, 1), ca.MX.sym('c0', 60, 1)], [output])


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
    print(state_dict.keys())

    # Função para converter arrays do NumPy para matrizes do CasADi (ca.DM)
    def to_casadi(tensor):
        return ca.DM(tensor.numpy())

    # Organizar os pesos e bias no formato esperado
    params = {
        'W_lstm': {
            'ih': to_casadi(state_dict['rnn_layer.weight_ih_l0']),  # Pesos entrada -> gates
            'hh': to_casadi(state_dict['rnn_layer.weight_hh_l0'])   # Pesos estado oculto -> gates
        },
        'b_lstm': {
            'ih': to_casadi(state_dict['rnn_layer.bias_ih_l0']),    # Bias entrada -> gates
            'hh': to_casadi(state_dict['rnn_layer.bias_hh_l0'])     # Bias estado oculto -> gates
        },
        'W_dense1': to_casadi(state_dict['dense_layers.0.weight']),  # Pesos da primeira camada densa
        'b_dense1': to_casadi(state_dict['dense_layers.0.bias']),    # Bias da primeira camada densa
        'W_dense2': to_casadi(state_dict['dense_layers.2.weight']),  # Pesos da segunda camada densa
        'b_dense2': to_casadi(state_dict['dense_layers.2.bias'])     # Bias da segunda camada densa
    }

    # Criar a instância da rede CasADi
    Modelo = CA_Model(params)

    # Construir x_seq a partir de y0 e u0
    x_seq = np.concatenate((y0, u0), axis=1)  # Formato (3, 4)
    x_seq = np.expand_dims(x_seq, axis=0)  # Formato (1, 3, 4)

    h0 = to_casadi(np.random.rand(60, 1))  # Estado oculto inicial
    c0 = to_casadi(np.random.rand(60, 1))

    saida =  Modelo.lstm_function(x_seq, h0, c0)

    print("Saída da rede CasADi:", saida)