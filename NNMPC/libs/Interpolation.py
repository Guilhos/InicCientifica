import numpy as np
import pandas as pd
import casadi as ca
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

class Interpolation:
    def __init__(self, file_path, decimal=','):
        self.file_path = file_path
        self.decimal = decimal
        self.params = [-25.0181, 42.0452, -17.9068, 3.0313]

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
        images_path = os.path.join("NNMPC", "images")
        os.makedirs(images_path, exist_ok=True)
        
        self.load_data()
        lut = self.interpolate()

        # Malhas de N_rot e Mass
        N_grid, M_grid = np.meshgrid(self.N_rot, self.Mass, indexing='ij')

        # Calcular Phi
        Phi_grid = np.zeros_like(N_grid)
        for i in range(N_grid.shape[0]):
            for j in range(N_grid.shape[1]):
                Phi_grid[i, j] = lut([N_grid[i, j], M_grid[i, j]])

        # Aplicar máscara: Phi > 0
        mask = Phi_grid > 0
        Phi_masked = np.where(mask, Phi_grid, np.nan)
        N_masked = np.where(mask, N_grid, np.nan)

        # Interpolação para plot: eixo X = Mass, eixo Y = Phi, contorno = N_rot
        from scipy.interpolate import griddata

        M_valid = M_grid[mask]
        Phi_valid = Phi_masked[mask]
        N_valid = N_masked[mask]

        mass_range = np.linspace(self.Mass.min(), self.Mass.max(), 200)
        phi_range = np.linspace(1, np.nanmax(Phi_valid), 200)
        M_mesh, Phi_mesh = np.meshgrid(mass_range, phi_range)
        N_interp = griddata((M_valid, Phi_valid), N_valid, (M_mesh, Phi_mesh), method='linear')

        # Calcular polinômio para a curva
        a0, a1, a2, a3 = self.params
        phi_curve = np.linspace(1.1, 2.3, 500)
        poly_curve = a0 + a1 * phi_curve + a2 * phi_curve**2 + a3 * phi_curve**3

        # Plotar
        fig, ax = plt.subplots(figsize=(10, 8))
        CS = ax.contour(M_mesh, Phi_mesh, N_interp, levels=20, colors='black')
        ax.clabel(CS, inline=True, fontsize=8, fmt="%.0f")

        # Plotar curva do polinômio
        ax.plot(poly_curve, phi_curve, 'r-', linewidth=2, label='Polinômio aplicado a Φ')

        ax.set_xlabel('Mass')
        ax.set_ylabel('Phi')
        ax.set_title('Curvas de Nível de N_rot e Curva Polinomial sobre Φ')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(images_path, "interpolation.png"))
        plt.show()
        
    def plot_with_trajectories(self, phi1=None, mass1=None, phi2=None, mass2=None):
        plt.rcParams.update({
        'font.size': 28,  # Aumenta o tamanho da fonte geral
        'axes.titlesize': 28,  # Tamanho do título dos eixos
        'axes.labelsize': 28,  # Tamanho dos rótulos dos eixos
        'xtick.labelsize': 28,  # Tamanho dos rótulos do eixo X
        'ytick.labelsize': 28,  # Tamanho dos rótulos do eixo Y
        'legend.fontsize': 28,  # Tamanho da fonte da legenda
        })

        images_path = os.path.join("NNMPC", "images")
        os.makedirs(images_path, exist_ok=True)
        
        self.load_data()
        lut = self.interpolate()

        # Malhas de N_rot e Mass
        N_grid, M_grid = np.meshgrid(self.N_rot, self.Mass, indexing='ij')

        # Calcular Phi
        Phi_grid = np.zeros_like(N_grid)
        for i in range(N_grid.shape[0]):
            for j in range(N_grid.shape[1]):
                Phi_grid[i, j] = lut([N_grid[i, j], M_grid[i, j]])

        # Aplicar máscara: Phi > 0
        mask = Phi_grid > 0
        Phi_masked = np.where(mask, Phi_grid, np.nan)
        N_masked = np.where(mask, N_grid, np.nan)

        from scipy.interpolate import griddata

        M_valid = M_grid[mask]
        Phi_valid = Phi_masked[mask]
        N_valid = N_masked[mask]

        mass_range = np.linspace(self.Mass.min(), self.Mass.max(), 200)
        phi_range = np.linspace(1, np.nanmax(Phi_valid), 200)
        M_mesh, Phi_mesh = np.meshgrid(mass_range, phi_range)
        N_interp = griddata((M_valid, Phi_valid), N_valid, (M_mesh, Phi_mesh), method='linear')

        # Curva polinomial
        a0, a1, a2, a3 = self.params
        phi_curve = np.linspace(1.1, 2.3, 500)
        poly_curve = a0 + a1 * phi_curve + a2 * phi_curve**2 + a3 * phi_curve**3

        # Plot
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.contour(M_mesh, Phi_mesh, N_interp, levels=20, colors='black', linewidth=2.5)

        ax.plot(poly_curve, phi_curve, color='gray', linestyle='-', linewidth=2, label='Restrição de surge', linewidth=2.5)

        # Trajetória 1 (se fornecida)
        if phi1 is not None and mass1 is not None:
            ax.plot(np.ravel(mass1), np.ravel(phi1), 'bo--', linewidth=2, markersize=4, label='RNN-MPC', linewidth=2.5)

        # Trajetória 2 (se fornecida)
        if phi2 is not None and mass2 is not None:
            ax.plot(np.ravel(mass2), np.ravel(phi2), 'rs--', linewidth=2, markersize=4, label='NNMPC', linewidth=2.5)

        ax.set_xlabel('Vazão / kg/s')
        ax.set_ylabel('Φ')
        ax.legend(
            loc='lower center',
            bbox_to_anchor=(0.5, -0.35),  # Posiciona a legenda abaixo do gráfico
            ncol=3,
            frameon=False
        )
        ax.set_ylim(1.2, 1.8)
        ax.set_xlim(6,11)
        plt.subplots_adjust(top=0.92, bottom=0.25, wspace=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(images_path, "interpolation_with_trajectories.png"))
    
if __name__ == "__main__":
    # Executar
    interp = Interpolation('NNMPC/libs/tabela_phi.csv')
    interp.plot()   

    