from CAMPC import Only_NMPC
from NNMPC import PINN_MPC
import pickle

qVazao_NN = 1/12.5653085708618164062**2
qPressao_NN = 0.01/9.30146217346191406250**2
rAlpha_NN = 1e-07/0.15**2
rN_NN = 5e-05/5000**2
qVazao_CA = 1/12.5653085708618164062**2
qPressao_CA = 0.05/9.30146217346191406250**2
rAlpha_CA = 1e-06/0.15**2
rN_CA = 5e-5/5000**2

p, m, q_NN, r_NN, q_CA, r_CA, steps = 12, 3, [qVazao_NN,qPressao_NN], [rAlpha_NN, rN_NN], [qVazao_CA,qPressao_CA], [rAlpha_CA, rN_CA], 3

NNMPC = PINN_MPC(p, m, q_NN, r_NN, steps)
CAMPC = Only_NMPC(p, m, q_CA, r_CA, steps)
iter_NN, Ymk_NN, Ypk_NN, Upk_NN, dURot_NN, dUAlpha_NN, YspM_NN, YspP_NN, YmMin_NN, Tempos_NN, PHI_NN = NNMPC.run()
#iter_CA, Ymk_CA, Ypk_CA, Upk_CA, dURot_CA, dUAlpha_CA, YspM_CA, YspP_CA, YmMin_CA, Tempos_CA, PHI_CA = CAMPC.run()

with open('NNMPC/libs/resultados_NNMPC.pkl', 'wb') as f:
    pickle.dump((iter_NN, Ymk_NN, Ypk_NN, Upk_NN, dURot_NN, dUAlpha_NN, YspM_NN, YspP_NN, YmMin_NN, Tempos_NN, PHI_NN), f)

#with open('NNMPC/libs/resultados_CAMPC.pkl', 'wb') as f:
    #pickle.dump((iter_CA, Ymk_CA, Ypk_CA, Upk_CA, dURot_CA, dUAlpha_CA, YspM_CA, YspP_CA, YmMin_CA, Tempos_CA, PHI_CA), f)