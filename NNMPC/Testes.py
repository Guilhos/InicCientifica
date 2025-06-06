import optuna
import numpy as np
import sys
import time
import pickle
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
                ISE_m * 5e2 / 12.5653085708618164062**2
                + ISE_p * 1e4 / 9.30146217346191406250**2
                + ISDNV_rot * 1 / 5000**2
                + ISDNV_alpha * 1 / 0.15**2
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
    
    # Reexecutar o melhor caso e salvar resultados
    best_params = study.best_params
    qVazao = 1 / 12.5653085708618164062**2
    qPressao = best_params['q_pressao_div'] / 9.30146217346191406250**2
    rAlpha = best_params['r_alpha_div'] / 0.15**2
    rN = best_params['r_n_div'] / 5000**2

    q = [qVazao, qPressao]
    r = [rAlpha, rN]

    p, m, steps = 12, 3, 3
    NNMPC = PINN_MPC(p, m, q, r, steps)
    iter_NN, Ymk_NN, Ypk_NN, Upk_NN, dURot_NN, dUAlpha_NN, YspM_NN, YspP_NN, YmMin_NN, Tempos_NN, PHI_NN = NNMPC.run()

    with open('NNMPC/libs/resultados_NNMPC.pkl', 'wb') as f:
        pickle.dump(
            (
                iter_NN, Ymk_NN, Ypk_NN, Upk_NN,
                dURot_NN, dUAlpha_NN, YspM_NN, YspP_NN,
                YmMin_NN, Tempos_NN, PHI_NN
            ), f
        )
        
    with open('NNMPC/libs/melhores_parametros.txt', 'w') as f:
        f.write("Melhores parâmetros encontrados:\n")
        for chave, valor in best_params.items():
            f.write(f"{chave}: {valor}\n")
        f.write(f"\nMenor erro total: {study.best_value}\n")
    

if __name__ == "__main__":
    run_optuna(total_trials=25)
