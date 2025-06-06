from libs.Interpolation import Interpolation
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

def superPlot(iter_NN, Ymk_NN,
              Ypk_NN, Upk_NN,
              YspM_NN, YspP_NN, YmMin_NN,
              Tempos_NN, iter_CA,
              Ymk_CA, Ypk_CA,
              Upk_CA, YspM_CA,
              YspP_CA, YmMin_CA, Tempos_CA, ISE, ISDMV):
    
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
        'font.size': 32,  # Aumenta o tamanho da fonte geral
        'axes.titlesize': 32,  # Tamanho do título dos eixos
        'axes.labelsize': 32,  # Tamanho dos rótulos dos eixos
        'xtick.labelsize': 32,  # Tamanho dos rótulos do eixo X
        'ytick.labelsize': 32,  # Tamanho dos rótulos do eixo Y
        'legend.fontsize': 32,  # Tamanho da fonte da legenda
    })

    # Função para configurar a legenda abaixo do gráfico
    def configurar_legenda():
        plt.legend(
            loc='lower center',
            bbox_to_anchor=(0.5, -0.5),  # Posiciona a legenda abaixo do gráfico
            ncol=3,
            frameon=False
        )

    # Vazão x Tempo
    plt.figure(figsize=(16,9), dpi = 400)
    plt.plot(x_NN / 2, np.array(Ypk_NN)[:, 0], label="RNN-MPC", color="blue", linewidth=2.5)
    plt.plot(x_CA / 2, np.array(Ypk_CA)[:, 0], linestyle="-", label = "NMPC", color="red", linewidth=2.5)
    plt.plot(x_NN / 2, YspM_NN.squeeze(), linestyle="-.", color="black", label="Set Point", linewidth=2.5)
    #plt.plot(x_NN / 2, YmMin_NN, linestyle="--", color="black", linewidth=2.5)
    plt.ylabel("Vazão / kg/s")
    plt.xlabel("Tempo / s")
    plt.grid()
    plt.ylim(6.5, 10)
    plt.legend(
            loc='lower center',
            bbox_to_anchor=(0.5, -0.35),  # Posiciona a legenda abaixo do gráfico
            ncol=3,
            frameon=False
        )
    plt.subplots_adjust(top=0.92, bottom=0.25, wspace=0.3)
    plt.savefig(os.path.join(images_path, "vazao_tempo_subplot.png"))
    
    # ESL (Erro entre o ponto da planta e o Limite Inferior)
    plt.figure(figsize=(16, 9), dpi = 400)
    erro_NN = np.array(Ypk_NN)[:, 0] - YmMin_NN.squeeze()
    plt.plot(x_NN / 2, erro_NN, label="RNN-MPC", color="blue", linewidth=2.5)
    erro_CA = np.array(Ypk_CA)[:, 0] - YmMin_CA.squeeze()
    plt.plot(x_CA / 2, erro_CA, linestyle="-", label="NMPC", color="red", linewidth=2.5)
    plt.axhline(0, color="black", linestyle="-.", label="Restrição de Surge", linewidth=2.5)
    plt.ylabel("Distância a margem de segurança / kg/s")
    plt.xlabel("Tempo / s")
    plt.grid()
    plt.ylim(-0.5, 1.2)  # Ajuste os limites do eixo Y conforme necessário
    plt.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, -0.35),  # Posiciona a legenda abaixo do gráfico
        ncol=3,
        frameon=False
    )
    plt.subplots_adjust(top=0.92, bottom=0.25, wspace=0.3)
    plt.savefig(os.path.join(images_path, "vazao_error_subplot.png"))

    # Pressão x Tempo
    plt.figure(figsize=(16,9), dpi = 400)
    plt.plot(x_NN / 2, np.array(Ypk_NN)[:, 1], label="RNN-MPC", color="blue", linewidth=2.5)
    plt.plot(x_CA / 2, np.array(Ypk_CA)[:, 1], linestyle="-", label="NMPC", color="red", linewidth=2.5)
    plt.plot(x_NN / 2, YspP_NN.squeeze(), linestyle="-.", color="black", label="Set Point", linewidth=2.5)
    plt.ylabel("Pressão / MPa")
    plt.xlabel("Tempo / s")
    plt.grid()
    plt.ylim(5.9, 7.65)
    plt.legend(
            loc='lower center',
            bbox_to_anchor=(0.5, -0.35),  # Posiciona a legenda abaixo do gráfico
            ncol=3,
            frameon=False
        )
    plt.subplots_adjust(top=0.92, bottom=0.25, wspace=0.3)
    plt.savefig(os.path.join(images_path, "pressao_tempo_subplot.png"))

    # Abertura da Válvula x Tempo
    plt.figure(figsize=(16,9), dpi = 400)
    plt.plot(x_NN / 2, np.array(Upk_NN)[:, 0], label="RNN-MPC", color="blue", linewidth=2.5)
    plt.plot(x_CA / 2, np.array(Upk_CA)[:, 0], linestyle="-", label="NMPC", color="red", linewidth=2.5)
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
    plt.figure(figsize=(16,9), dpi = 400)
    plt.plot(x_NN / 2, np.array(Upk_NN)[:, 1] // 60, label="RNN-MPC", color="blue", linewidth=2.5)
    plt.plot(x_CA / 2, np.array(Upk_CA)[:, 1] // 60, linestyle="-", label="NMPC", color="red", linewidth=2.5)
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

    # Filtrar os dados para ignorar outliers acima de 5 segundos
    Tempos_CA_filtrado = [t for t in Tempos_CA if t <= 5]
    Tempos_NN_filtrado = [t for t in Tempos_NN if t <= 5]

    # Histograma das Frequências de Tempo
    plt.figure(figsize=(16, 9), dpi=400)
    plt.hist(Tempos_CA_filtrado, bins=26, color='red', alpha=0.7, edgecolor='black', label="NMPC")
    plt.axvline(np.mean(Tempos_CA_filtrado), color='red', linestyle="--", label="Média NMPC", linewidth=2.5)
    print(f"Média NMPC (sem outliers): {np.mean(Tempos_CA_filtrado):.2f} s")

    plt.hist(Tempos_NN_filtrado, bins=26, color='blue', alpha=0.7, edgecolor='black', label="RNN-MPC")
    plt.axvline(np.mean(Tempos_NN_filtrado), color='blue', linestyle="--", label="Média RNN-MPC", linewidth=2.5)
    print(f"Média PIRNN-MPC (sem outliers): {np.mean(Tempos_NN_filtrado):.2f} s")

    plt.axvspan(0, 0.5, color='black', alpha=0.3, label="Zona de Interesse")  # Região cinza claro
    plt.axvline(0.5, color='green', linestyle="--", label="Tempo Amostral", linewidth=2.5)

    plt.xlabel("Tempo / s")
    plt.ylabel("Frequência")
    configurar_legenda()
    plt.tight_layout(h_pad=3.0)
    plt.subplots_adjust(top=0.92, bottom=0.30)
    plt.savefig(os.path.join(images_path, "histograma_frequencias_tempo.png"))
    plt.close()

    
def calcular_ISDMV(sinal_controle):
    sinal_controle = np.array(sinal_controle).flatten()
    isdmv = np.sum(sinal_controle** 2)
    return isdmv

def calcular_ISE(referencia, saida):
    erro = np.array(referencia) - np.array(saida)
    ise = np.sum(erro**2)
    return ise

interp = Interpolation('NNMPC/libs/tabela_phi.csv')
# Carregando
with open('NNMPC/libs/resultados_NNMPC.pkl', 'rb') as f:
    iter_NN, Ymk_NN, Ypk_NN, Upk_NN, dURot_NN, dUAlpha_NN, YspM_NN, YspP_NN, YmMin_NN, Tempos_NN, PHI_NN = pickle.load(f)

with open('NNMPC/libs/resultados_CAMPC.pkl', 'rb') as f:
    iter_CA, Ymk_CA, Ypk_CA, Upk_CA, dURot_CA, dUAlpha_CA, YspM_CA, YspP_CA, YmMin_CA, Tempos_CA, PHI_CA = pickle.load(f)

ISE_NN_M = calcular_ISE(YspM_NN, np.array(Ypk_NN)[:, 0])
ISE_NN_P = calcular_ISE(YspP_NN, np.array(Ypk_NN)[:, 1])
ISE_CA_M = calcular_ISE(YspM_CA, np.array(Ypk_CA)[:, 0])
ISE_CA_P = calcular_ISE(YspP_CA, np.array(Ypk_CA)[:, 1])

ISE = [ISE_NN_M, ISE_NN_P, ISE_CA_M, ISE_CA_P]
print(f"ISE_NN_M: {ISE_NN_M:.2f}\n ISE_NN_P: {ISE_NN_P:.2f}\n ISE_CA_M: {ISE_CA_M:.2f}\n ISE_CA_P: {ISE_CA_P:.2f}")

ISDMV_CA_A = calcular_ISDMV(dUAlpha_CA)
ISDMV_CA_N = calcular_ISDMV(dURot_CA)
ISDMV_NN_A = calcular_ISDMV(dUAlpha_NN)
ISDMV_NN_N = calcular_ISDMV(dURot_NN)

ISDMV = [ISDMV_NN_A, ISDMV_NN_N, ISDMV_CA_A, ISDMV_CA_N]
print(f"ISDMV_NN_A: {ISDMV_NN_A:.2f}\n ISDMV_NN_N: {ISDMV_NN_N:.2f}\n ISDMV_CA_A: {ISDMV_CA_A:.2f}\n ISDMV_CA_N: {ISDMV_CA_N:.2f}")

print(f'ISE - Vazão & ${ISE_NN_M:.2e}$ & ${ISE_CA_M:.2e}$ & {((ISE_NN_M - ISE_CA_M)/ISE_CA_M)*100:.2f}\\% \\\\ \n'
      f'ISE - Pressão & ${ISE_NN_P:.2e}$ & ${ISE_CA_P:.2e}$ & {((ISE_NN_P - ISE_CA_P)/ISE_CA_P)*100:.2f}\\% \\\\ \n'
      f'ISDMV - Válvula & ${ISDMV_NN_A:.2e}$ & ${ISDMV_CA_A:.2e}$ & {((ISDMV_NN_A - ISDMV_CA_A)/ISDMV_CA_A)*100:.2f}\\% \\\\ \n'
      f'ISDMV - Vel. Rotação & ${ISDMV_NN_N:.2e}$ & ${ISDMV_CA_N:.2e}$ & {((ISDMV_NN_N - ISDMV_CA_N)/ISDMV_CA_N)*100:.2f}\\% \\\\')

print(f'Tempo Médio - RNN-MPC: {np.mean(Tempos_NN):.2f} s\n'
      f'Tempo Médio - NMPC: {np.mean(Tempos_CA):.2f} s\n'
      f'Variação de Tempo: {(np.mean(Tempos_CA)/np.mean(Tempos_NN)):.2f} vezes mais rápido\n')

superPlot(iter_NN, Ymk_NN,
          Ypk_NN, Upk_NN,
          YspM_NN, YspP_NN, YmMin_NN,
          Tempos_NN, iter_CA,
          Ymk_CA, Ypk_CA,
          Upk_CA, YspM_CA,
          YspP_CA, YmMin_CA, Tempos_CA, ISE, ISDMV)

phi1 = np.array(PHI_NN)
mass1 = np.array(Ypk_NN)[:, 0]
phi2 = np.array(PHI_CA)
mass2 = np.array(Ypk_CA)[:, 0]
interp.plot_with_trajectories(phi1=phi1, mass1=mass1, phi2=phi2, mass2=mass2)