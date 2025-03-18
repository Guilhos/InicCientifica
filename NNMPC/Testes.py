import numpy as np
import casadi as ca

def matriz_triangular_identidade(m, n, N):
    """
    Cria uma matriz triangular inferior onde os elementos não nulos são matrizes identidade de tamanho N.
    
    Parâmetros:
    m, n : int - Número de blocos na matriz
    N : int - Tamanho das matrizes identidade
    
    Retorna:
    Matriz CasADi DM de tamanho (m*N, n*N)
    """
    matriz = np.zeros((m * N, n * N))
    
    for i in range(m):
        for j in range(n):
            if j <= i:
                matriz[i * N:(i + 1) * N, j * N:(j + 1) * N] = np.eye(N)
    
    return ca.DM(matriz)  # Convertendo para CasADi DM

# Exemplo de uso
m, n, N = 3, 3, 2
M = matriz_triangular_identidade(m, n, N)

# Criando uma variável simbólica compatível
x = ca.MX.sym('x', n*N, 1)

# Multiplicação com CasADi
resultado = M @ x

print("Matriz Triangular Inferior (CasADi):")
print(M)

print("Expressão simbólica resultante:")
print(resultado)
