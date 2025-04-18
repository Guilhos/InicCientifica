from CAMPC import Only_NMPC
from NNMPC import PINN_MPC
import numpy as np
import matplotlib.pyplot as plt
import os

def superPlot(iter_NN, Ymk_NN,
              Ypk_NN, Upk_NN,
              YspM_NN, YspP_NN,
              Tempos_NN, iter_CA,
              Ymk_CA, Ypk_CA,
              Upk_CA, YspM_CA,
              YspP_CA, Tempos_CA):
    
    images_path = os.path.join("NNMPC", "images")
    os.makedirs(images_path, exist_ok=True)

    x_NN = np.linspace(0, iter_NN, iter_NN)
    x_CA = np.linspace(0, iter_CA, iter_CA)
    YspM_NN = np.array(YspM_NN)
    YspP_NN = np.array(YspP_NN)
    YspM_CA = np.array(YspM_CA)
    YspP_CA = np.array(YspP_CA)

    # Vazão x Tempo
    plt.figure(figsize=(12,10))
    plt.subplot(2, 1, 1)
    plt.plot(x_NN / 2, np.array(Ymk_NN)[:, 0], label="Modelo - Rede Neural", color='green')
    plt.plot(x_NN / 2, np.array(Ypk_NN)[:, 0], label="Planta - Rede Neural", color="blue")
    plt.plot(x_NN / 2, YspM_NN.squeeze(), linestyle="--", color="red", label="y_sp")
    plt.plot([0, iter_NN / 2], [3.5, 3.5], linestyle="--", color="black")
    plt.plot([0, iter_NN / 2], [12.3, 12.3], linestyle="--", color="black")
    plt.title("Vazão x Tempo (NN)")
    plt.ylabel("Vazão / kg/s")
    plt.xlabel("Tempo / s")
    plt.legend()
    plt.grid()
    plt.ylim(3, 12.8)

    plt.subplot(2, 1, 2)
    plt.plot(x_CA / 2, np.array(Ymk_CA)[:, 0], label="Modelo - Nominal", color='green')
    plt.plot(x_CA / 2, np.array(Ypk_CA)[:, 0], label="Planta - Nominal", color="blue")
    plt.plot(x_CA / 2, YspM_CA.squeeze(), linestyle="--", color="red", label="y_sp")
    plt.plot([0, iter_CA / 2], [3.5, 3.5], linestyle="--", color="black")
    plt.plot([0, iter_CA / 2], [12.3, 12.3], linestyle="--", color="black")
    plt.title("Vazão x Tempo (CA)")
    plt.ylabel("Vazão / kg/s")
    plt.xlabel("Tempo / s")
    plt.legend()
    plt.grid()
    plt.ylim(3, 12.8)
    plt.tight_layout(h_pad=3.0)
    plt.savefig(os.path.join(images_path, "vazao_tempo.png"))
    plt.show()
    plt.close()

    # Pressão x Tempo
    plt.figure(figsize=(12,10))
    plt.subplot(2, 1, 1)
    plt.plot(x_NN / 2, np.array(Ymk_NN)[:, 1], label="Modelo - Rede Neural", color="green")
    plt.plot(x_NN / 2, np.array(Ypk_NN)[:, 1], label="Planta - Rede Neural", color="blue")
    plt.plot(x_NN / 2, YspP_NN.squeeze(), linestyle="--", color="red", label="y_sp")
    plt.plot([0, iter_NN / 2], [5.27, 5.27], linestyle="--", color="black")
    plt.plot([0, iter_NN / 2], [9.3, 9.3], linestyle="--", color="black")
    plt.title("Pressão x Tempo (NN)")
    plt.ylabel("Pressão / kPa")
    plt.xlabel("Tempo / s")
    plt.legend()
    plt.grid()
    plt.ylim(4.77, 9.83)

    plt.subplot(2, 1, 2)
    plt.plot(x_CA / 2, np.array(Ymk_CA)[:, 1], label="Modelo - Nominal", color="green")
    plt.plot(x_CA / 2, np.array(Ypk_CA)[:, 1], label="Planta - Nominal", color="blue")
    plt.plot(x_CA / 2, YspP_CA.squeeze(), linestyle="--", color="red", label="y_sp")
    plt.plot([0, iter_CA / 2], [5.27, 5.27], linestyle="--", color="black")
    plt.plot([0, iter_CA / 2], [9.3, 9.3], linestyle="--", color="black")
    plt.title("Pressão x Tempo (CA)")
    plt.ylabel("Pressão / kPa")
    plt.xlabel("Tempo / s")
    plt.legend()
    plt.grid()
    plt.ylim(4.77, 9.83)
    plt.tight_layout(h_pad=3.0)
    plt.savefig(os.path.join(images_path, "pressao_tempo.png"))
    plt.show()
    plt.close()

    # Abertura da Válvula x Tempo
    plt.figure(figsize=(12,10))
    plt.subplot(2, 1, 1)
    plt.plot(x_NN / 2, np.array(Upk_NN)[:, 0], label="Abertura - Rede Neural", color="blue")
    plt.plot([0, iter_NN / 2], [0.35, 0.35], linestyle="--", color="black")
    plt.plot([0, iter_NN / 2], [0.65, 0.65], linestyle="--", color="black")
    plt.title("Abertura da Válvula x Tempo (NN)")
    plt.ylabel("Alpha / %")
    plt.xlabel("Tempo / s")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(x_CA / 2, np.array(Upk_CA)[:, 0], label="Abertura - Nominal", color="blue")
    plt.plot([0, iter_CA / 2], [0.35, 0.35], linestyle="--", color="black")
    plt.plot([0, iter_CA / 2], [0.65, 0.65], linestyle="--", color="black")
    plt.title("Abertura da Válvula x Tempo (CA)")
    plt.ylabel("Alpha / %")
    plt.xlabel("Tempo / s")
    plt.legend()
    plt.grid()
    plt.tight_layout(h_pad=3.0)
    plt.savefig(os.path.join(images_path, "abertura_valvula_tempo.png"))
    plt.show()
    plt.close()

    # Velocidade de Rotação x Tempo
    plt.figure(figsize=(12,10))
    plt.subplot(2, 1, 1)
    plt.plot(x_NN / 2, np.array(Upk_NN)[:, 1], label="Vel. Rotação - Rede Neural", color="blue")
    plt.plot([0, iter_NN / 2], [27e3, 27e3], linestyle="--", color="black")
    plt.plot([0, iter_NN / 2], [5e4, 5e4], linestyle="--", color="black")
    plt.title("Velocidade de Rotação x Tempo (NN)")
    plt.ylabel("N / Hz")
    plt.xlabel("Tempo / s")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(x_CA / 2, np.array(Upk_CA)[:, 1], label="Vel. Rotação - Nominal", color="blue")
    plt.plot([0, iter_CA / 2], [27e3, 27e3], linestyle="--", color="black")
    plt.plot([0, iter_CA / 2], [5e4, 5e4], linestyle="--", color="black")
    plt.title("Velocidade de Rotação x Tempo (CA)")
    plt.ylabel("N / Hz")
    plt.xlabel("Tempo / s")
    plt.legend()
    plt.grid()
    plt.tight_layout(h_pad=3.0)
    plt.savefig(os.path.join(images_path, "velocidade_rotacao_tempo.png"))
    plt.show()
    plt.close()

    # Tempo por Iteração
    plt.figure(figsize=(12,10))
    plt.subplot(2, 1, 1)
    plt.plot(x_NN, Tempos_NN, color="blue")
    plt.plot([0, iter_NN], [0.5, 0.5], linestyle="--", color="black")
    plt.plot([0, iter_NN], [np.mean(Tempos_NN), np.mean(Tempos_NN)], linestyle="--", color="red", label=f"Média: {np.mean(Tempos_NN):.2f} s")
    plt.title("Tempo por Iteração (NN)")
    plt.ylabel("Tempo / s")
    plt.xlabel("Iteração")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(x_CA, Tempos_CA, color="blue")
    plt.plot([0, iter_CA], [0.5, 0.5], linestyle="--", color="black")
    plt.plot([0, iter_CA], [np.mean(Tempos_CA), np.mean(Tempos_CA)], linestyle="--", color="red", label=f"Média: {np.mean(Tempos_CA):.2f} s")
    plt.title("Tempo por Iteração (CA)")
    plt.ylabel("Tempo / s")
    plt.xlabel("Iteração")
    plt.legend()
    plt.grid()
    plt.tight_layout(h_pad=3.0)
    plt.savefig(os.path.join(images_path, "tempo_iteracao.png"))
    plt.show()
    plt.close()

    # Histograma das Frequências de Tempo
    plt.figure(figsize=(12,10))
    plt.hist(Tempos_CA, bins=20, color='red', alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(Tempos_CA), color='red', linestyle="--", label=f"Média CA: {np.mean(Tempos_CA):.2f} s")
    plt.hist(Tempos_NN, bins=20, color='blue', alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(Tempos_NN), color='blue', linestyle="--", label=f"Média NN: {np.mean(Tempos_NN):.2f} s")
    plt.title("Histograma das Frequências de Tempo")
    plt.xlabel("Tempo")
    plt.ylabel("Frequência")
    plt.legend()
    plt.tight_layout(h_pad=3.0)
    plt.savefig(os.path.join(images_path, "histograma_frequencias_tempo.png"))
    plt.show()
    plt.close()
    
qVazao = 1/12.5653085708618164062**2
qPressao = 0.1/9.30146217346191406250**2
rAlpha = 0/0.15**2
rN = 1e-4/5000**2

p, m, q, r, steps = 12, 3, [qVazao,qPressao], [rAlpha, rN], 3
NNMPC = PINN_MPC(p, m, q, r, steps)
CAMPC = Only_NMPC(p, m, q, r, steps)
iter_NN, Ymk_NN, Ypk_NN, Upk_NN, YspM_NN, YspP_NN, Tempos_NN =  NNMPC.run()
iter_CA, Ymk_CA, Ypk_CA, Upk_CA, YspM_CA, YspP_CA, Tempos_CA =  CAMPC.run()

superPlot(iter_NN, Ymk_NN,
          Ypk_NN, Upk_NN,
          YspM_NN, YspP_NN,
          Tempos_NN, iter_CA,
          Ymk_CA, Ypk_CA,
          Upk_CA, YspM_CA,
          YspP_CA, Tempos_CA)