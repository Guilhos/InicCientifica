import torch
import onnx
import numpy as np
import time
import onnxruntime as ort

class NN_Model:
    def __init__(self, x,, p, m):
        self.x = x # Vetor de variáveis
        self.p = p # Horizonte de Predição
        self.m = m # Horizonte de Controle

        self.onnx_model_path = "./modelo.onnx"
        self.onnx_session = ort.InferenceSession(self.onnx_model_path)
        self.input_tensor = torch.zeros((1, 3, 4), dtype=torch.float32)

    def run(self):
        vazaoMassica = [self.x[0, 0].item(), self.x[1, 0].item(), self.x[2, 0].item()]
        pressaoPlenum = [self.x[0, 1].item(), self.x[1, 1].item(), self.x[2, 1].item()]

        for i in range(p):
            # Atualizar os valores do tensor diretamente
            input_tensor[0, :, 0] = torch.tensor(vazaoMassica[-3:])
            input_tensor[0, :, 1] = torch.tensor(pressaoPlenum[-3:])
            input_tensor[0, :, 2] = self.x[i, :, 2]
            input_tensor[0, :, 3] = self.x[i, :, 3]

            # Previsão com desativação do gradiente
            onnx_inputs = {'input': input_tensor.numpy()}
            onnx_outputs = onnx_session.run(None, onnx_inputs)

            # Adicionar previsões diretamente
            massFlowrate_pred.append(onnx_outputs[0][0, 0, 0])
            PlenumPressure_pred.append(onnx_outputs[0][0, 0, 1])
