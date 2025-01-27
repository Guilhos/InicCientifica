import numpy as np
import torch
import onnxruntime as ort
import matplotlib.pyplot as plt

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
