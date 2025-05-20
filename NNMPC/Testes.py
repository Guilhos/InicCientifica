import optuna
import numpy as np
import sys
import time
from CAMPC import Only_NMPC
from NNMPC import PINN_MPC  # se usado

def calcular_ISDMV(sinal_controle):
    sinal_controle = np.array(sinal_controle).flatten()
    return np.sum(sinal_controle**2)

def calcular_ISE(referencia, saida):
    erro = np.array(referencia) - np.array(saida)
    return np.sum(erro**2)

def progress_bar(atual, total, erro=None):
    largura = 30
    preenchido = int(largura * atual / total)
    barra = "#" * preenchido + "-" * (largura - preenchido)
    texto = f"[{barra}] Trial {atual}/{total}"
    if erro is not None:
        texto += f" | Erro: {erro:.4f}"
    sys.stdout.write("\r" + texto)
    sys.stdout.flush()

def run_optuna(total_trials):
    def objective(trial):
        q_pressao_div = trial.suggest_float('q_pressao_div', 1e-6, 1, log=True)
        r_alpha_div = trial.suggest_float('r_alpha_div', 1e-6, 1, log=True)
        r_n_div = trial.suggest_float('r_n_div', 1e-6, 1, log=True)

        qVazao = 1 / 12.5653085708618164062**2
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

            erro_total = (
                ISE_m * 1e6 / 12.5653085708618164062**2
                + ISE_p * 1e7 / 9.30146217346191406250**2
                + ISDNV_rot * 1e-6 / 5000**2
                + ISDNV_alpha * 1e-3 / 0.15**2
            )
        except Exception as e:
            erro_total = float('inf')

        progresso_atual = len(study.trials) + 1
        progress_bar(progresso_atual, total_trials, erro_total)
        return erro_total

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=total_trials)

    print("\n✅ Otimização concluída!")
    print("Melhores parâmetros encontrados:", study.best_params)
    print("Menor erro total:", study.best_value)

if __name__ == "__main__":
    run_optuna(total_trials=50)
