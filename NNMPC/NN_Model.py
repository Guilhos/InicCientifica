import numpy as np
import torch
import onnxruntime as ort
import matplotlib.pyplot as plt

class NN_Model:
    def __init__(self, p=50, m = 3):
        self.p = p # Horizonte de Predição
        self.m = m # Horizonte de Controle

        self.onnx_model_path = "./libs/modelo.onnx"
        self.onnx_session = ort.InferenceSession(self.onnx_model_path)
        self.input_tensor = torch.zeros((1, 3, 4), dtype=torch.float32)
        self.vazaoMassica = []
        self.pressaoPlenum = []

    def run(self,x,du):
        self.vazaoMassica = [x[0, 0].item(), x[1, 0].item(), x[2, 0].item()]
        self.pressaoPlenum = [x[0, 1].item(), x[1, 1].item(), x[2, 1].item()]
        nu = len(du) / self.m
        du = np.concatenate((np.array(du), np.zeros(p - len(np.array(du)))))

        for k in range(self.p):
            # Atualizar os valores do tensor diretamente
            self.input_tensor[0, :, 0] = torch.tensor(self.vazaoMassica[-3:])
            self.input_tensor[0, :, 1] = torch.tensor(self.pressaoPlenum[-3:])
            self.input_tensor[0, :, 2] = torch.tensor(x[:, 2])
            self.input_tensor[0, :, 3] = torch.tensor(x[:, 3])
            #print(x[-1, 2], x[-1, 3])
            # Previsão com desativação do gradiente
            onnx_inputs = {'input': self.input_tensor.numpy()}
            onnx_outputs = self.onnx_session.run(None, onnx_inputs)
            print(k)
            duk = du[k:k+2]
            print(duk)
            # Adicionar previsões diretamente
            x[-3, 2] = x[-2, 2]
            x[-2, 2] = x[-1, 2]
            x[-1, 2] = x[-1, 2] + duk[0]

            x[-3, 3] = x[-3, 3]
            x[-2, 3] = x[-1, 3]
            x[-1, 3] = x[-1, 3] + duk[1]

            self.vazaoMassica.append(onnx_outputs[0][0, 0, 0])
            self.pressaoPlenum.append(onnx_outputs[0][0, 0, 1])
        print(self.du)
            
if __name__ == '__main__':
    from libs.Interpolation import Interpolation
    from libs.simulationn import Simulation

    p = 50
    m = 3
    du = [0.05,2000,-0.02,-1000,0.1,1000]

    lut = Interpolation('./libs/tabela_phi.csv')
    lut.load_data()
    interpolation = lut.interpolate()

    sim = Simulation(interpolation)
    sim.run()

    NNModel = NN_Model(p,m)
    NNModel.run(sim.output,du)
    print(f'{NNModel.vazaoMassica}\n{NNModel.pressaoPlenum}')

    pltz = np.linspace(1,p+3, p+3)
    fig, (ax1,ax2) = plt.subplots(2,1)
    ax1.plot(pltz,NNModel.pressaoPlenum, color='r')
    ax2.plot(pltz, NNModel.vazaoMassica)
    fig.show()
