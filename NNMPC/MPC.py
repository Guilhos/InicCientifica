import numpy as np
import matplotlib.pyplot as plt
from libs.simulationn import Simulation
from libs.Interpolation import Interpolation
from NN_Model import NN_Model

lut = Interpolation('./tabela_phi.csv')
lut.load_data()
interpolation = lut.interpolate()
