%%% MPC
%% Mapa do Compressor
sheet_url = 'https://docs.google.com/spreadsheets/d/1r6tQcEkqFMsYhwhe4K3eQ-eAxf3cMnnECnsVwvJnxus/export?format=csv';
opts = detectImportOptions(sheet_url);  
opts = setvartype(opts,'double');       
data = readmatrix(sheet_url, opts);
N_rot = 2e4:1e3:6e4;
Mass = 3:0.1:21;
Phi = data';
[NR_grid, Mass_grid] = ndgrid(N_rot, Mass);
lut = @(n,m) interp2(Mass_grid, NR_grid, Phi, m, n, 'spline');

%% Constantes do Sistema
A1 = (2.6)*(1e-3); % Área do duto
Lc = 2; % Tamanho do duto
kv = 0.38; % Constante de válvula
P1 = 4.5; % Pressão de entrada
P_out = 5; % Pressão de saída
C = 479; % Velocidade do som no gás
Vp = 2; % 

%% Ponto de operação
m_exp = 7.745;
p_exp = 6.662;
a_exp = 0.5;
n_exp = 3.85;

%% Linearização do sistema
% dm/dt = a11 * m + a12 * p + b12 * n
a11 = P1 * A1 / Lc * 1e6 * (lut(n_exp*1e4, m_exp + 0.1) - lut(n_exp*1e4, m_exp - 0.1)) / 2 / 0.1;
a12 = -A1 / Lc * 1e6;
b12 = P1 * A1 / Lc * 1e10 * (lut(n_exp*1e4 + 1e3, m_exp) - lut(n_exp*1e4 - 1e3, m_exp)) / 2 / 1e3;

% dp/dt = a21 * m + a22 * p + b21 * a
a21 = C^2 / Vp / 1e6;
a22 = -C^2 / Vp * a_exp * kv * 500 / sqrt((p_exp - P_out) * 1000) / 1e6;
b21 = -C^2 / Vp * kv * sqrt((p_exp - P_out) * 1000) / 1e6;

%% Sistema continuo
Ac = [a11,a12;
    a21,a22];
Bc = [0,b12;
    b21,0];
Cc = eye(2);
Dc = zeros(2,2);

Ts = 0.5;
sys_c = ss(Ac, Bc, Cc, Dc);

%% Sistema discreto
sys_d = c2d(sys_c, Ts, 'zoh');

[A, B, C, D] = ssdata(sys_d);

%% Configurações do controlador
p = 12;
m = 3;
nX = 2;
nU = 2;

%% Controle com setpoint
A_ = [A, B;
      zeros(nU, size(A,2)), eye(nU)];

B_ = [B;
      eye(nU)]; 

Qtil = [2,2];
Rtil = [2,2];

Q = diag(kron(ones(1,p), Qtil));
R = diag(kron(ones(1,m), Rtil));

C_ = [Cc, zeros(nU,nU)];
Cu = [zeros(nX,nX), eye(nU)];

psi = [C_ * A_];
for i = 0:(p-1)
    psi = [psi; psi(end-1:end, :) * A_];
end

theta = zeros(nX*p,nU*m);
for i = 0:p
    for j = 0:(i+1)
        if (j+1)*nU <= nU*m
            theta(i*nU+1:(i+1)*Nu, j*nU+1:(j+1)*nU) = C_ * (A_^ (i-j)) * B_;
        else
            continue;
        end
    end
end

psiU = [Cu * A_]
for i = 0:p
    psiU = [psiU; psiU(end-1:end,:)*A_]
end

thetaU = zeros(nX*p,nU*m)
for i = 0:p
    for j = 0:(i+1)
        if (j+1)*nU <= nU*m
            theta(i*nU+1:(i+1)*Nu, j*nU+1:(j+1)*nU) = Cu * (A_^ (i-j)) * B_;
        else
            continue
        end
    end
end

H = theta' * Q * theta + R;
H = (H + H') * 0.5;
F = [psi' * Q * theta; kron(ones(p,1), eye(Nx))' * -Q * theta];

G = [theta;
    -theta;
    thetaU;
    -thetaU
    eye(nX*p,nU*m);
    -eye(nX*p,nU*m)];
Su = [-psi, zeros(nX*p,nX);
    psi, zeros(nX*p,nX);
    -psiU, zeros(nX*p,nX);
    psiU, zeros(nX*p,nX);
    zeros(nX*p,nX*nU), zeros(nX*p,nX);
    zeros(nX*p,nX*nU), zeros(nX*p,nX)];

%% Simulação do MPC
K = 100;
x_op = [m_exp;
        p_exp];
u_op = [a_exp; 
        n_exp];
y_op = x_op;

x0 = [7.745; 6.662] - x_op;
u0 = [0.5; 3.85] - u_op;

xk = x0;
uk = u0;

x_k = [xk;uk];
yspk = [7.745; 6.662] - y_op;
z_k = [x_k', yspk'];

yMax = repmat([12.3; 9.3] - y_op, p, 1);
yMin = repmat([5.5; 5.27] - y_op, p, 1);
uMax = repmat([0.65; 5] - u_op, p, 1);
uMin = repmat([0.35; 2.7] - u_op, p, 1);
deltaUMax = repmat([0.1; 0.25], p, 1);
deltaUMin = repmat([-0.1; -0.25], p, 1);

w = [yMax;
     -yMin;
     uMax;
     -uMin;
     deltaUMax;
     -deltaUMin];

deltaU_value = zeros(K, nU*m);
deltaU_mpc = zeros(K, nU);
z_k_store_mpc = zeros(K, nX + nX + nU);
