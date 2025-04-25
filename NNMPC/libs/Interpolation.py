import numpy as np
import pandas as pd
import casadi as ca
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Interpolation:
    def __init__(self, file_path, decimal=','):
        self.file_path = file_path
        self.decimal = decimal
        self.params = [-4.117976349902422e+01, 7.819497805337005e+01,
                       -4.406691928857834e+01, 8.836867650811681e+00]

    def load_data(self):
        # Simulando leitura de dados (substitua pelo seu CSV real)
        self.data = pd.read_csv(self.file_path, decimal=self.decimal)
        self.N_rot = np.arange(2e4, 6e4, 1e3)  # Shape: (40,)
        self.Mass = np.arange(3, 21.1, 0.1)    # Shape: (181,)
        # Simulando Phi como uma matriz 40x181 (substitua pelos dados reais)
        self.Phi = self.data.values   # Exemplo; use self.data.values para dados reais

    def interpolate(self):
        phi_flat = self.Phi.ravel(order='F')
        lut = ca.interpolant('name', 'bspline', [self.N_rot, self.Mass], phi_flat)
        return lut

    def plot(self):
        self.load_data()
        lut = self.interpolate()

        # Criar malha para N_rot e Mass
        N_grid, M_grid = np.meshgrid(self.N_rot, self.Mass, indexing='ij')

        # Avaliar Phi usando a interpolação
        Phi_grid = np.zeros_like(N_grid)
        for i in range(N_grid.shape[0]):
            for j in range(N_grid.shape[1]):
                Phi_grid[i, j] = lut([N_grid[i, j], M_grid[i, j]])

        # Calcular a superfície polinomial sobre Phi interpolado
        a0, a1, a2, a3 = self.params
        Poly_grid = a0 + a1 * Phi_grid + a2 * Phi_grid**2 + a3 * Phi_grid**3

        # Máscara para manter apenas valores com Phi > 0
        mask = Phi_grid > 0
        N_masked = np.where(mask, N_grid, np.nan)
        M_masked = np.where(mask, M_grid, np.nan)
        Phi_masked = np.where(mask, Phi_grid, np.nan)
        Poly_masked = np.where(mask, Poly_grid, np.nan)

        # Identificar cruzamento aproximado: onde |Phi - Poly(Phi)| < tolerância
        diff = np.abs(Phi_grid - Poly_grid)
        tolerance = 1e-2
        cross_mask = (diff < tolerance) & mask
        N_cross = N_grid[cross_mask]
        M_cross = M_grid[cross_mask]
        Z_cross = Phi_grid[cross_mask]  # ou Poly_grid[cross_mask], são semelhantes
        
        # Printar os valores da linha de cruzamento
        print("=== Linha de Cruzamento (Phi ≈ Poly(Phi)) ===")
        for n, m, z in zip(N_cross, M_cross, Z_cross):
            print(f"N_rot: {n:.2f}, Mass: {m:.2f}, Phi: {z:.6f}")

        # Criar figura 3D
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Superfície interpolada (Phi)
        ax.plot_surface(N_masked, M_masked, Phi_masked, cmap='viridis', alpha=0.7)

        # Superfície polinomial
        ax.plot_surface(N_masked, M_masked, Poly_masked, cmap='plasma', alpha=0.5)

        # Linha de cruzamento
        ax.plot(N_cross, M_cross, Z_cross, 'r-', linewidth=2, label='Cruzamento Phi = Poly(Phi)')

        # Rótulos
        ax.set_xlabel('N_rot')
        ax.set_ylabel('Mass')
        ax.set_zlabel('Phi / Polinômio')
        ax.set_title('Phi Interpolado vs Polinômio (Phi > 0)')
        ax.legend()

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Executar
    interp = Interpolation('NNMPC/libs/tabela_phi.csv')
    interp.plot()   

    