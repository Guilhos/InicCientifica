import numpy as np

p = 50 # Horizonte de predição
m = 3 # Horizonte de controle
q = 0.1 # Peso Q
r =  1 # Peso R
nY = 2
nU = 2

u_min, u_max = np.array([[0.35],[27e3]]), np.array([[0.65],[5e4]])  # Limites na variável manipulada
dU_min, dU_max = np.array([[0.01], [500]]), np.array([[0.15], [5000]])  # Limites nos incrementos de u
y_min, y_max = np.array([[3.5],[5.27]]), np.array([[12.3],[10.33]]) # Limites na variável controlada

# Setpoint provisório
y_sp = (y_max + y_min)/2 
print(y_sp)

def iTil(x, n):
    n = np.tile(x, (n,1))
    return n

dU_min = iTil(dU_min, m)
print(dU_min.shape)
dU_min = np.concatenate((dU_min,np.zeros((int(nU) * (p-m), 1))))
print(dU_min.shape)
dU_min = dU_min.T
print(dU_min.shape)

def diagMatrix(x,n):
    x = np.float64(x)
    X_matrix = np.full((n,n),0, dtype=np.float64)
    np.fill_diagonal(X_matrix,x)
    return X_matrix

r = diagMatrix(r,nU*m)
print(r.shape)
print(r)