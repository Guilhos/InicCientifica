from CAMPC import Only_NMPC
from NNMPC import PINN_MPC
import pickle

qVazao = 0.1/12.5653085708618164062**2
qPressao = 0.1/9.30146217346191406250**2
rAlpha = 0/0.15**2
rN = 1e-4/5000**2

p, m, q, r, steps = 12, 3, [qVazao,qPressao], [rAlpha, rN], 3
NNMPC = PINN_MPC(p, m, q, r, steps)
CAMPC = Only_NMPC(p, m, q, r, steps)
iter_NN, Ymk_NN, Ypk_NN, Upk_NN, dURot_NN, dUAlpha_NN, YspM_NN, YspP_NN, YmMin_NN, Tempos_NN, PHI_NN =  NNMPC.run()
iter_CA, Ymk_CA, Ypk_CA, Upk_CA, dURot_CA, dUAlpha_CA, YspM_CA, YspP_CA, YmMin_CA, Tempos_CA, PHI_CA = CAMPC.run()

with open('NNMPC/libs/resultados_NNMPC.pkl', 'wb') as f:
    pickle.dump((iter_NN, Ymk_NN, Ypk_NN, Upk_NN, dURot_NN, dUAlpha_NN, YspM_NN, YspP_NN, YmMin_NN, Tempos_NN, PHI_NN), f)

with open('NNMPC/libs/resultados_CAMPC.pkl', 'wb') as f:
    pickle.dump((iter_CA, Ymk_CA, Ypk_CA, Upk_CA, dURot_CA, dUAlpha_CA, YspM_CA, YspP_CA, YmMin_CA, Tempos_CA, PHI_CA), f)