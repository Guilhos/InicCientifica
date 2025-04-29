from CAMPC import Only_NMPC
from NNMPC import PINN_MPC
from libs.Interpolation import Interpolation
import numpy as np
import matplotlib.pyplot as plt
import os

def superPlot(iter_NN, Ymk_NN,
              Ypk_NN, Upk_NN,
              YspM_NN, YspP_NN, YmMin_NN,
              Tempos_NN, iter_CA,
              Ymk_CA, Ypk_CA,
              Upk_CA, YspM_CA,
              YspP_CA, YmMin_CA, Tempos_CA, ISE, ISDNV):
    
    images_path = os.path.join("NNMPC", "images")
    os.makedirs(images_path, exist_ok=True)

    x_NN = np.linspace(0, iter_NN, iter_NN)
    x_CA = np.linspace(0, iter_CA, iter_CA)
    YspM_NN = np.array(YspM_NN)
    YspP_NN = np.array(YspP_NN)
    YspM_CA = np.array(YspM_CA)
    YspP_CA = np.array(YspP_CA)
    YmMin_NN = np.array(YmMin_NN)
    YmMin_CA = np.array(YmMin_CA)

    # Atualiza os parâmetros de estilo
    plt.rcParams.update({
        'font.size': 28,  # Aumenta o tamanho da fonte geral
        'axes.titlesize': 28,  # Tamanho do título dos eixos
        'axes.labelsize': 28,  # Tamanho dos rótulos dos eixos
        'xtick.labelsize': 28,  # Tamanho dos rótulos do eixo X
        'ytick.labelsize': 28,  # Tamanho dos rótulos do eixo Y
        'legend.fontsize': 28,  # Tamanho da fonte da legenda
    })

    # Função para configurar a legenda abaixo do gráfico
    def configurar_legenda():
        plt.legend(
            loc='lower center',
            bbox_to_anchor=(0.5, -0.45),  # Posiciona a legenda abaixo do gráfico
            ncol=3,
            frameon=False
        )

    # Vazão x Tempo
    plt.figure(figsize=(20,9))
    plt.plot(x_NN / 2, np.array(Ypk_NN)[:, 0], label="RNN-MPC", color="blue", linewidth=2.5)
    plt.plot(x_CA / 2, np.array(Ypk_CA)[:, 0], linestyle="--", label = "NMPC", color="red", linewidth=2.5)
    plt.plot(x_NN / 2, YspM_NN.squeeze(), linestyle="-.", color="black", label="Set Point", linewidth=2.5)
    #plt.plot(x_NN / 2, YmMin_NN, linestyle="--", color="black", linewidth=2.5)
    plt.ylabel("Vazão / kg/s")
    plt.xlabel("Tempo / s")
    plt.grid()
    plt.ylim(4.5, 12.8)
    plt.legend(
            loc='lower center',
            bbox_to_anchor=(0.5, -0.4),  # Posiciona a legenda abaixo do gráfico
            ncol=3,
            frameon=False
        )
    plt.subplots_adjust(top=0.92, bottom=0.25, wspace=0.3)
    plt.savefig(os.path.join(images_path, "vazao_tempo_subplot.png"))
    
    # ESL (Erro entre o ponto da planta e o Limite Inferior)
    plt.figure(figsize=(20, 9))
    erro_NN = np.array(Ypk_NN)[:, 0] - YmMin_NN.squeeze()
    plt.plot(x_NN / 2, erro_NN, label="RNN-MPC", color="blue", linewidth=2.5)
    erro_CA = np.array(Ypk_CA)[:, 0] - YmMin_CA.squeeze()
    plt.plot(x_CA / 2, erro_CA, linestyle="--", label="NMPC", color="red", linewidth=2.5)
    plt.axhline(0, color="black", linestyle="-.", linewidth=1.5, label="Linha de Surge", linewidth=2.5)
    plt.ylabel("Distância a linha de surge / kg/s")
    plt.xlabel("Tempo / s")
    plt.grid()
    plt.ylim(-1, 1.5)  # Ajuste os limites do eixo Y conforme necessário
    plt.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, -0.4),  # Posiciona a legenda abaixo do gráfico
        ncol=2,
        frameon=False
    )
    plt.subplots_adjust(top=0.92, bottom=0.25, wspace=0.3)
    plt.savefig(os.path.join(images_path, "vazao_error_subplot.png"))

    # Pressão x Tempo
    plt.figure(figsize=(20,9))
    plt.plot(x_NN / 2, np.array(Ypk_NN)[:, 1], label="RNN-MPC", color="blue", linewidth=2.5)
    plt.plot(x_CA / 2, np.array(Ypk_CA)[:, 1], linestyle="--", label="NMPC", color="red", linewidth=2.5)
    plt.plot(x_NN / 2, YspP_NN.squeeze(), linestyle="-.", color="black", label="Set Point", linewidth=2.5)
    plt.ylabel("Pressão / kPa")
    plt.xlabel("Tempo / s")
    plt.grid()
    plt.ylim(4.77, 9.8)
    plt.legend(
            loc='lower center',
            bbox_to_anchor=(0.5, -0.4),  # Posiciona a legenda abaixo do gráfico
            ncol=3,
            frameon=False
        )
    plt.subplots_adjust(top=0.92, bottom=0.25, wspace=0.3)
    plt.savefig(os.path.join(images_path, "pressao_tempo_subplot.png"))

    # Abertura da Válvula x Tempo
    plt.figure(figsize=(20,9))
    plt.plot(x_NN / 2, np.array(Upk_NN)[:, 0], label="RNN-MPC", color="blue", linewidth=2.5)
    plt.plot(x_CA / 2, np.array(Upk_CA)[:, 0], linestyle="--", label="NMPC", color="red", linewidth=2.5)
    plt.plot([0, iter_NN / 2], [0.35, 0.35], linestyle="-.", color="black", label="Restrições de Contorno", linewidth=2.5)
    plt.plot([0, iter_NN / 2], [0.65, 0.65], linestyle="-.", color="black", linewidth=2.5)
    plt.ylabel("Alpha / %")
    plt.xlabel("Tempo / s")
    plt.grid()
    plt.legend(
            loc='lower center',
            bbox_to_anchor=(0.5, -0.35),  # Posiciona a legenda abaixo do gráfico
            ncol=3,
            frameon=False
        )
    plt.subplots_adjust(top=0.92, bottom=0.25, wspace=0.3)
    plt.savefig(os.path.join(images_path, "abertura_valvula_tempo_subplot.png"))

    # Subplots para Velocidade de Rotação x Tempo
    fig, axs = plt.subplots(1, 2, figsize=(32,9), sharey=True)

    # Velocidade de Rotação x Tempo
    plt.figure(figsize=(20,9))
    plt.plot(x_NN / 2, np.array(Upk_NN)[:, 1] // 60, label="RNN-MPC", color="blue", linewidth=2.5)
    plt.plot(x_CA / 2, np.array(Upk_CA)[:, 1] // 60, linestyle="--", label="NMPC", color="red", linewidth=2.5)
    plt.plot([0, iter_NN / 2], [27e3 // 60, 27e3 // 60], linestyle="-.", color="black", label="Restrições de Contorno", linewidth=2.5)
    plt.plot([0, iter_NN / 2], [5e4 // 60, 5e4 // 60], linestyle="-.", color="black", linewidth=2.5)
    plt.ylabel("N / Hz")
    plt.xlabel("Tempo / s")
    plt.grid()
    plt.legend(
            loc='lower center',
            bbox_to_anchor=(0.5, -0.35),  # Posiciona a legenda abaixo do gráfico
            ncol=3,
            frameon=False
        )
    plt.subplots_adjust(top=0.92, bottom=0.25, wspace=0.3)
    plt.savefig(os.path.join(images_path, "velocidade_rotacao_tempo_subplot.png"))

    # Histograma das Frequências de Tempo
    plt.figure(figsize=(20,9))
    plt.hist(Tempos_CA, bins=26, color='red', alpha=0.7, edgecolor='black', label="NMPC")
    plt.axvline(np.mean(Tempos_CA), color='red', linestyle="--", label="Média NMPC")
    print(f"Média NMPC: {np.mean(Tempos_CA):.2f} s")
    plt.hist(Tempos_NN, bins=26, color='blue', alpha=0.7, edgecolor='black', label ="PIRNN-MPC")
    plt.axvline(np.mean(Tempos_NN), color='blue', linestyle="--", label="Média RNN-MPC")
    print(f"Média PIRNN-MPC: {np.mean(Tempos_NN):.2f} s")
    plt.axvline(0.5, color='green', linestyle="--", label="Tempo Amostral")
    plt.axvspan(0, 0.5, color='black', alpha=0.3, label="Região de Interesse")  # Adiciona a região cinza claro
    plt.xlabel("Tempo / s")
    plt.ylabel("Frequência")
    configurar_legenda()
    plt.tight_layout(h_pad=3.0)
    plt.subplots_adjust(top=0.92, bottom=0.30)
    plt.savefig(os.path.join(images_path, "histograma_frequencias_tempo.png"))
    plt.close()
    
qVazao = 0.1/12.5653085708618164062**2
qPressao = 0.1/9.30146217346191406250**2
rAlpha = 0/0.15**2
rN = 1e-4/5000**2

interp = Interpolation('NNMPC/libs/tabela_phi.csv')

p, m, q, r, steps = 12, 3, [qVazao,qPressao], [rAlpha, rN], 3
NNMPC = PINN_MPC(p, m, q, r, steps)
CAMPC = Only_NMPC(p, m, q, r, steps)
iter_NN, Ymk_NN, Ypk_NN, Upk_NN, dURot_NN, dUAlpha_NN, YspM_NN, YspP_NN, YmMin_NN, Tempos_NN, PHI_NN =  NNMPC.run()
iter_CA, Ymk_CA, Ypk_CA, Upk_CA, dURot_CA, dUAlpha_CA, YspM_CA, YspP_CA, YmMin_CA, Tempos_CA, PHI_CA = CAMPC.run()

def calcular_ISE(referencia, saida):
    erro = np.array(referencia) - np.array(saida)
    ise = np.sum(erro**2)
    return ise

ISE_NN_M = calcular_ISE(YspM_NN, np.array(Ypk_NN)[:, 0])
ISE_NN_P = calcular_ISE(YspP_NN, np.array(Ypk_NN)[:, 1])
ISE_CA_M = calcular_ISE(YspM_CA, np.array(Ypk_CA)[:, 0])
ISE_CA_P = calcular_ISE(YspP_CA, np.array(Ypk_CA)[:, 1])

ISE = [ISE_NN_M, ISE_NN_P, ISE_CA_M, ISE_CA_P]
print(f"ISE_NN_M: {ISE_NN_M:.2f}\n ISE_NN_P: {ISE_NN_P:.2f}\n ISE_CA_M: {ISE_CA_M:.2f}\n ISE_CA_P: {ISE_CA_P:.2f}")

def calcular_ISDNV(sinal_controle):
    sinal_controle = np.array(sinal_controle).flatten()
    derivada = np.diff(sinal_controle)
    isdnv = np.sum((derivada / 0.1) ** 2)
    return isdnv

ISDNV_CA_M = calcular_ISDNV(dUAlpha_CA)
ISDNV_CA_P = calcular_ISDNV(dURot_CA)
ISDNV_NN_M = calcular_ISDNV(dUAlpha_NN)
ISDNV_NN_P = calcular_ISDNV(dURot_NN)

ISDNV = [ISDNV_NN_M, ISDNV_NN_P, ISDNV_CA_M, ISDNV_CA_P]
print(f"ISDNV_NN_M: {ISDNV_NN_M:.2f}\n ISDNV_NN_P: {ISDNV_NN_P:.2f}\n ISDNV_CA_M: {ISDNV_CA_M:.2f}\n ISDNV_CA_P: {ISDNV_CA_P:.2f}")

superPlot(iter_NN, Ymk_NN,
          Ypk_NN, Upk_NN,
          YspM_NN, YspP_NN, YmMin_NN,
          Tempos_NN, iter_CA,
          Ymk_CA, Ypk_CA,
          Upk_CA, YspM_CA,
          YspP_CA, YmMin_CA, Tempos_CA, ISE, ISDNV)

phi1 = np.array(PHI_NN)
mass1 = np.array(Ypk_NN)[:, 0]
phi2 = np.array(PHI_CA)
mass2 = np.array(Ypk_CA)[:, 0]
interp.plot_with_trajectories(phi1=phi1, mass1=mass1, phi2=phi2, mass2=mass2)