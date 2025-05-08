import optuna
import numpy as np
import tkinter as tk
from tkinter import ttk
from threading import Thread
from CAMPC import Only_NMPC
from NNMPC import PINN_MPC

def calcular_ISDMV(sinal_controle):
    sinal_controle = np.array(sinal_controle).flatten()
    isdmv = np.sum(sinal_controle** 2)
    return isdmv

def calcular_ISE(referencia, saida):
    erro = np.array(referencia) - np.array(saida)
    ise = np.sum(erro**2)
    return ise

class OptunaWithProgress:
    def __init__(self, total_trials):
        self.total_trials = total_trials
        self.completed_trials = 0

        # Criando a janela com tkinter
        self.root = tk.Tk()
        self.root.title("Otimização com Optuna")
        self.root.geometry("400x100")

        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(self.root, maximum=total_trials, length=300, variable=self.progress_var)
        self.progress.pack(pady=20)

        self.label = tk.Label(self.root, text="Iniciando...")
        self.label.pack()

        # Rodar a otimização em uma thread separada
        thread = Thread(target=self.run_optimization)
        thread.start()

        # Inicia o loop da GUI
        self.root.mainloop()

    def update_progress(self):
        self.completed_trials += 1
        self.progress_var.set(self.completed_trials)
        self.label.config(text=f"Trial {self.completed_trials}/{self.total_trials}")
        self.root.update_idletasks()

    def run_optimization(self):
        def objective(trial):
            # Sugere os dividendos
            q_vazao_div = trial.suggest_float('q_vazao_div', 0.001, 100, log=True)
            q_pressao_div = trial.suggest_float('q_pressao_div', 0.001, 100, log=True)
            r_alpha_div = trial.suggest_float('r_alpha_div', 0.01, 10, log=True)
            r_n_div = trial.suggest_float('r_n_div', 0.01, 10000, log=True)

            # Calcula Q e R
            qVazao = q_vazao_div / 12.5653085708618164062**2
            qPressao = q_pressao_div / 9.30146217346191406250**2
            rAlpha = r_alpha_div / 0.15**2
            rN = r_n_div / 5000**2

            q = [qVazao, qPressao]
            r = [rAlpha, rN]

            p, m, steps = 12, 3, 3
            try:
                NNMPC = PINN_MPC(p, m, q, r, steps)
                iter_NN, Ymk_NN, Ypk_NN, Upk_NN, dURot_NN, dUAlpha_NN, YspM_NN, YspP_NN, YmMin_NN, Tempos_NN, PHI_NN = NNMPC.run()

                YspM = YspM_NN
                YspP = YspP_NN
                Ypk = np.array(Ypk_NN)

                ISE_m = calcular_ISE(YspM, Ypk[:, 0])
                ISE_p = calcular_ISE(YspP, Ypk[:, 1])
                ISDNV_rot = calcular_ISDMV(dURot_NN)
                ISDNV_alpha = calcular_ISDMV(dUAlpha_NN)

                erro_total = ISE_m + 10 * ISE_p + ISDNV_rot * 1e-4 + ISDNV_alpha * 1e5
                return erro_total

            except Exception as e:
                print("Erro ao executar NNMPC:", e)
                return float('inf')
            finally:
                self.update_progress()

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.total_trials)

        print("Melhores parâmetros encontrados:", study.best_params)
        print("Menor erro total:", study.best_value)

        self.label.config(text="Otimização concluída!")

# Executar com 50 trials
OptunaWithProgress(total_trials=25)
