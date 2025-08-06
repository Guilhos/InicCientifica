import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
from torchdeq import get_deq
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import StandardScaler
torch.manual_seed(42)

# --- Carregamento de Dados para o novo sistema ---
sheet_url="https://docs.google.com/spreadsheets/d/1UPq-_KwH0DZXkwSryfIc5ePeHTOPsQVSuMVaHeimwSI/edit#gid=0"
url_1=sheet_url.replace('/edit#gid=', '/export?format=csv&gid=')
arquivo=pd.read_csv(url_1,decimal=',')
arquivo.head()

# --- Preparação dos Dados (U=Q, Y=H) ---
t = np.array(arquivo['t'],dtype='float64')
Q = np.array(arquivo['Qi(t)'],dtype='float64').reshape(-1, 1)
H = np.array(arquivo['H(t)'],dtype='float64').reshape(-1, 1)

# --- NOVA ETAPA: NORMALIZAÇÃO DE DADOS PARA ESTABILIDADE ---
# É crucial para evitar gradientes explosivos e NaN.
# Os scalers são treinados APENAS nos dados de treino para evitar data leakage.
train_split_idx = int(len(Q) * 0.6)

q_scaler = StandardScaler()
h_scaler = StandardScaler()

# Fit nos dados de treino
q_scaler.fit(Q[:train_split_idx])
h_scaler.fit(H[:train_split_idx])

# Transformar todos os dados
Q_scaled = q_scaler.transform(Q)
H_scaled = h_scaler.transform(H)

# --- PREPARAÇÃO DE DADOS AUTORREGRESSIVOS COM DADOS NORMALIZADOS ---
# O objetivo é prever H_scaled(t+1) a partir de [Q_scaled(t), H_scaled(t)]
U_autoregressive = np.hstack([Q_scaled[:-1], H_scaled[:-1]])
Y_autoregressive = H_scaled[1:]

# Converte para tensores
U_tensor = torch.tensor(U_autoregressive, dtype=torch.float32)
Y_tensor = torch.tensor(Y_autoregressive, dtype=torch.float32)

# --- Divisão de Dados Sequencial ---
total_size = len(U_tensor) 
train_size = int(total_size * 0.60) 
val_size = int(total_size * 0.2) 
test_size = total_size - train_size - val_size 

U_train, Y_train = U_tensor[:train_size], Y_tensor[:train_size] 
U_val, Y_val = U_tensor[train_size:train_size + val_size], Y_tensor[train_size:train_size + val_size] 
U_test, Y_test = U_tensor[train_size + val_size:], Y_tensor[train_size + val_size:]

# A variável de tempo para o teste precisa corresponder aos dados de teste
t_original_test_idx_start = train_size + val_size + 1
t_test = t[t_original_test_idx_start : t_original_test_idx_start + len(U_test)]

print("Dados normalizados e preparados para modelo autorregressivo.")
print(f"Dimensões do Treino (U, Y): {U_train.shape}, {Y_train.shape}")
print(f"Dimensões da Validação (U, Y): {U_val.shape}, {Y_val.shape}")
print(f"Dimensões do Teste (U, Y): {U_test.shape}, {Y_test.shape}")

# --- Definição do Modelo DEQ ---
class SystemDEQ(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # Rede f(z, u) 
        self.f_network = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.deq = get_deq(f_solver='broyden', f_max_iter=100, f_tol=1e-6)

    def forward(self, u):
        f = lambda z: self.f_network(torch.cat([z,u], dim=-1))
        z0 = torch.zeros(u.shape[0], self.f_network[0].in_features - u.shape[1], device=u.device)
        z_star = self.deq(f,z0)[0][0]
        y_pred = self.output_layer(z_star)
        return y_pred
    
# --- Hiperparâmetros ---
INPUT_DIM = U_train.shape[1]
HIDDEN_DIM = 32
OUTPUT_DIM = Y_train.shape[1]
LEARNING_RATE = 5e-4 # Taxa de aprendizado reduzida para maior estabilidade
EPOCHS = 500

# --- Instanciar o modelo, a função de perda e o otimizador ---
model = SystemDEQ(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM)
print("\nEstrutura do Modelo:")
print(model)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = StepLR(optimizer, step_size=200, gamma=0.1)

# --- Listas para armazenar perdas ---
train_losses = []
val_losses = []

# --- Loop de Treinamento ---
for epoch in range(EPOCHS):
    model.train()
    Y_pred_train = model(U_train)
    loss = criterion(Y_pred_train, Y_train)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            Y_pred_val = model(U_val)
            val_loss = criterion(Y_pred_val, Y_val)
            train_losses.append(loss.item())
            val_losses.append(val_loss.item())
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}")

# --- PREDIÇÃO AUTORREGRESSIVA (CLOSED-LOOP) ---
model.eval()
predictions_scaled = []
with torch.no_grad():
    # Initialize the simulation with the first H value from the dataset
    current_h_scaled = U_test[0, 1].view(1, 1)
    
    # Run the simulation for the entire length of the dataset
    for i in range(len(U_test)):
        # Use the true Q input from the dataset
        current_q_scaled = U_test[i, 0].view(1, 1)
        model_input = torch.cat([current_q_scaled, current_h_scaled], dim=1)
        
        # Predict the next H
        next_h_scaled = model(model_input)
        
        predictions_scaled.append(next_h_scaled.item())
        # Feed the prediction back into the model for the next step
        current_h_scaled = next_h_scaled

# Este tensor agora contém as previsões para o conjunto de teste (200 passos)
Y_simulation_scaled = torch.tensor(predictions_scaled).view(-1, 1)

# --- INVERSÃO DA NORMALIZAÇÃO para cálculo da perda e plotagem ---
# Isso torna os resultados interpretáveis na escala original

# Inverte a escala das previsões da simulação. 
# Esta variável já é a sua previsão final para o teste.
Y_pred_test_inversed = h_scaler.inverse_transform(Y_simulation_scaled.numpy())

# Inverte a escala dos alvos reais do conjunto de teste (Y_test)
Y_test_original_scale = h_scaler.inverse_transform(Y_test.numpy())

# A linha incorreta que fatiaca o array foi removida.

# Calcula a perda na escala original usando apenas os dados de teste
# Agora as formas são compatíveis: (200, 1) e (200, 1)
test_loss = np.mean((Y_pred_test_inversed - Y_test_original_scale)**2)
print(f'\nPerda final no teste (MSE na escala original): {test_loss:.6f}')
# --- Plotagem ---
images_path = os.path.join("SuspensaoAtiva", "images")
os.makedirs(images_path, exist_ok=True)

# 1. Gráfico da Perda de Treino vs. Validação
plt.figure(figsize=(12, 5))
plt.plot(train_losses, label='Perda de Treino (Train Loss)')
plt.plot(val_losses, label='Perda de Validação (Validation Loss)')
plt.title('Curva de Aprendizagem: Perda de Treino vs. Validação')
plt.xlabel('Épocas (x10)')
plt.ylabel('Erro Quadrático Médio (Normalizado)')
plt.legend()
plt.grid(True)
plt.savefig("MPC_to_NN/DEQ - Shaojie Bai/Losses.png")

# Get the original Q values for the test set for plotting
Q_test_original = q_scaler.inverse_transform(U_test[:, 0].numpy().reshape(-1, 1))

# 2. Gráfico da Dinâmica do Sistema (Original vs. Previsto) na escala original
plt.figure(figsize=(14, 7))
# Now all arrays have the same length (200)
plt.plot(t_test, Y_test_original_scale, 'b-', label='H Original (Teste)')
plt.plot(t_test, Y_pred_test_inversed, 'r--', label='H Previsto (Autoregressivo)')
plt.plot(t_test, Q_test_original, 'g-.', label='Q Entrada (Teste)')
plt.title('Comparação da Dinâmica do Sistema: Predição Autoregressiva (Escala Original)')
plt.xlabel('Tempo (t)')
plt.ylabel('Valor')
plt.legend()
plt.grid(True)
plt.savefig("MPC_to_NN/DEQ - Shaojie Bai/Previsto.png")