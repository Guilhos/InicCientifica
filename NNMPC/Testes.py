import casadi as ca
import onnxruntime as ort
import numpy as np
from libs.simulationn import Simulation

def build_model_mpc(y, u, dU, P):
    """
    Constrói a função simbólica CasADi usando o modelo ONNX.
    Entradas:
      - y: vetor de estados iniciais (y0)
      - u: vetor de controles iniciais (u0)
      - dU: variação de controle ao longo do horizonte de predição
      - P: horizonte de predição
    Saída:
      - y_pred: vetor de previsões de saída ao longo do horizonte P
    """
    # Carregar modelo ONNX
    onnx_model_path = "NNMPC/libs/modelo.onnx"
    onnx_session = ort.InferenceSession(onnx_model_path)
    
    # Criar variáveis CasADi
    y_casadi = ca.MX(y)
    u_casadi = ca.MX(u)
    dU_casadi = ca.MX(dU)
    
    # Inicializar listas para armazenar previsões
    vazaoMassica = [y_casadi[0], y_casadi[2], y_casadi[4]]
    pressaoPlenum = [y_casadi[1], y_casadi[3], y_casadi[5]]
    alpha = [u_casadi[0], u_casadi[2], u_casadi[4]]
    N_Rot = [u_casadi[1], u_casadi[3], u_casadi[5]]
    
    # Loop para calcular previsões ao longo de P
    for k in range(P):
        
        # Criar entrada para a rede ONNX
        input_tensor = np.array([
            [vazaoMassica[-3], pressaoPlenum[-3], alpha[-3], N_Rot[-3]],
            [vazaoMassica[-2], pressaoPlenum[-2], alpha[-2], N_Rot[-2]],
            [vazaoMassica[-1], pressaoPlenum[-1], alpha[-1], N_Rot[-1]]
        ], dtype=np.float32).reshape(1, 3, 4)
        
        # Rodar o modelo ONNX
        onnx_inputs = {'input': input_tensor}
        onnx_outputs = onnx_session.run(None, onnx_inputs)
        
        # Obter saída da rede
        pred_vazao = onnx_outputs[0][0, 0, 0]
        pred_pressao = onnx_outputs[0][0, 0, 1]
        
        # Atualizar estados
        dUk = dU_casadi[2*k:2*(k+1)]
        alpha.append(alpha[-1] + dUk[0])
        N_Rot.append(N_Rot[-1] + dUk[1])
        vazaoMassica.append(pred_vazao)
        pressaoPlenum.append(pred_pressao)
    
    # Gerar vetor de saída
    y_pred = ca.vertcat(*vazaoMassica[3:], *pressaoPlenum[3:])
    
    # Criar função CasADi
    f_mpc = ca.Function('f_mpc', [y_casadi, u_casadi, dU_casadi], [y_pred])
    
    return f_mpc

sim = Simulation(3,3)
y0, u0 = sim.pIniciais()
dU = [[0],[0],[0],[0],[0],[0]]

func = build_model_mpc(ca.DM(y0),ca.DM(u0),ca.DM(dU), 50)

saida = func(ca.DM(y0),ca.DM(u0),ca.DM(dU))
print(saida)