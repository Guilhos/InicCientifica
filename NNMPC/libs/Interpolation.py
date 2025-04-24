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
        # self.data = pd.read_csv(self.file_path, decimal=self.decimal)
        self.N_rot = np.arange(2e4, 6e4, 1e3)  # Shape: (40,)
        self.Mass = np.arange(3, 21.1, 0.1)    # Shape: (181,)
        # Simulando Phi como uma matriz 40x181 (substitua pelos dados reais)
        self.Phi = np.random.rand(40, 181)  # Exemplo; use self.data.values para dados reais

    def interpolate(self):
        phi_flat = self.Phi.ravel(order='F')
        lut = ca.interpolant('name', 'bspline', [self.N_rot, self.Mass], phi_flat)
        return lut

    def plot(self):
        # Carregar dados e criar interpolante
        self.load_data()
        lut = self.interpolate()

        # Fixar N_rot (usando o valor médio, por exemplo)
        n_rot_fixed = np.mean(self.N_rot)

        # Avaliar a interpolação para todos os valores de Mass
        phi_values = np.zeros_like(self.Mass)
        for i, mass in enumerate(self.Mass):
            phi_values[i] = lut([n_rot_fixed, mass])

        # Plotar Mass vs Phi (interpolação)
        plt.figure(figsize=(8, 6))
        plt.plot(self.Mass, phi_values, 'b-', label='Phi Interpolado')

        # Plotar a curva polinomial
        phi_range = np.linspace(np.min(phi_values), np.max(phi_values), 181)
        poly = (self.params[0] + self.params[1] * phi_range +
                self.params[2] * phi_range**2 + self.params[3] * phi_range**3)
        plt.plot(self.Mass, poly, 'r--', label='Curva Polinomial')

        # Configurações do gráfico
        plt.xlabel('Mass')
        plt.ylabel('Phi')
        plt.title('Phi Interpolado e Curva Polinomial vs Mass')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    # Executar
    interp = Interpolation('dummy_path.csv')
    interp.plot()   

    