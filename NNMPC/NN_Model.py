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
        self.vazaoMassica = []
        self.pressaoPlenum = []

    def run(self,y,u,dU):
        self.vazaoMassica = [y[0, 0].item(), y[1, 0].item(), y[2, 0].item()]
        self.pressaoPlenum = [y[0, 1].item(), y[1, 1].item(), y[2, 1].item()]
        nU = len(dU) / self.m
        dU = np.concatenate((np.array(dU), np.zeros(int(nU)*p - len(np.array(dU)))))

        for k in range(self.p):
            # Atualizar os valores do tensor diretamente
            self.input_tensor[0, :, 0] = torch.tensor(self.vazaoMassica[-3:])
            self.input_tensor[0, :, 1] = torch.tensor(self.pressaoPlenum[-3:])
            self.input_tensor[0, :, 2] = torch.tensor(u[:, 0])
            self.input_tensor[0, :, 3] = torch.tensor(u[:, 1])
            # Previsão com desativação do gradiente
            onnx_inputs = {'input': self.input_tensor.numpy()}
            onnx_outputs = self.onnx_session.run(None, onnx_inputs)
            dUk = dU[2*k:2*(k+1)]
            
            # Adicionar previsões diretamente
            u[-3, 0] = u[-2, 0]
            u[-2, 0] = u[-1, 0]
            u[-1, 0] = u[-1, 0] + dUk[0]

            u[-3, 1] = u[-2, 1]
            u[-2, 1] = u[-1, 1]
            u[-1, 1] = u[-1, 1] + dUk[1]

            self.vazaoMassica.append(onnx_outputs[0][0, 0, 0])
            self.pressaoPlenum.append(onnx_outputs[0][0, 0, 1])
        
        return (self.vazaoMassica[3:], self.pressaoPlenum[3:])
            
if __name__ == '__main__':
    from libs.simulationn import Simulation

    p = 50
    m = 3
    dU = [0.05,2000,-0.02,-1000,0.1,1000]

    sim = Simulation()
    y0, u0 = sim.run()

    NNModel = NN_Model(p,m)
    y = NNModel.run(y0,u0,dU)
    
    

    pltz = np.linspace(1,p, p)
    fig, (ax1,ax2) = plt.subplots(2,1)
    ax1.plot(pltz, y[0], color='r')
    ax2.plot(pltz, y[1])
    plt.show()
