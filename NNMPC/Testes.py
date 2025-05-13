import optuna
import numpy as np
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QProgressBar
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from CAMPC import Only_NMPC
from NNMPC import PINN_MPC

def calcular_ISDMV(sinal_controle):
    sinal_controle = np.array(sinal_controle).flatten()
    return np.sum(sinal_controle**2)

def calcular_ISE(referencia, saida):
    erro = np.array(referencia) - np.array(saida)
    return np.sum(erro**2)

class OptunaThread(QThread):
    progresso = pyqtSignal(int, int)
    finalizado = pyqtSignal(dict, float)

    def __init__(self, total_trials):
        super().__init__()
        self.total_trials = total_trials

    def run(self):
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
                NNMPC = Only_NMPC(p, m, q, r, steps)
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
                return erro_total

            except Exception as e:
                print("Erro ao executar NNMPC:", e)
                return float('inf')
            finally:
                self.progresso.emit(len(self.study.trials), self.total_trials)

        self.study = optuna.create_study(direction='minimize')
        self.study.optimize(objective, n_trials=self.total_trials)
        self.finalizado.emit(self.study.best_params, self.study.best_value)

class OptunaApp(QWidget):
    def __init__(self, total_trials):
        super().__init__()
        self.setWindowTitle("Otimização com Optuna")
        self.setGeometry(100, 100, 400, 100)

        self.layout = QVBoxLayout()
        self.label = QLabel("Iniciando...")
        self.label.setAlignment(Qt.AlignCenter)
        self.progress = QProgressBar()
        self.progress.setMaximum(total_trials)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.progress)
        self.setLayout(self.layout)

        self.thread = OptunaThread(total_trials)
        self.thread.progresso.connect(self.atualizar_progresso)
        self.thread.finalizado.connect(self.finalizar)
        self.thread.start()

    def atualizar_progresso(self, atual, total):
        self.progress.setValue(atual)
        self.label.setText(f"Trial {atual}/{total}")

    def finalizar(self, best_params, best_value):
        self.label.setText("Otimização concluída!")
        print("Melhores parâmetros encontrados:", best_params)
        print("Menor erro total:", best_value)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OptunaApp(total_trials=20)
    window.show()
    sys.exit(app.exec_())
