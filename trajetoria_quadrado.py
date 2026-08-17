import numpy as np
from scipy.io import savemat

# ============================================================
# TRAJETÓRIA QUADRADA NO PLANO XY
# Colunas:
# [x, y, z, vx, vy, vz, theta_z, theta_y, theta_x]
# ============================================================

N = 400

# ------------------------------------------------------------
# POSIÇÃO
# ------------------------------------------------------------

# Trecho 1: (0.2, 0.0) -> (0.2, 0.2)
x1 = np.full(50, 0.2)
y1 = np.linspace(0.0, 0.2, 50)
z1 = np.zeros(50)

# Trecho 2: (0.2, 0.2) -> (-0.2, 0.2)
x2 = np.linspace(0.2, -0.2, 100)
y2 = np.full(100, 0.2)
z2 = np.zeros(100)

# Trecho 3: (-0.2, 0.2) -> (-0.2, -0.2)
x3 = np.full(100, -0.2)
y3 = np.linspace(0.2, -0.2, 100)
z3 = np.zeros(100)

# Trecho 4: (-0.2, -0.2) -> (0.2, -0.2)
x4 = np.linspace(-0.2, 0.2, 100)
y4 = np.full(100, -0.2)
z4 = np.zeros(100)

# Trecho 5: (0.2, -0.2) -> (0.2, 0.0)
x5 = np.full(50, 0.2)
y5 = np.linspace(-0.2, 0.0, 50)
z5 = np.zeros(50)

x = np.concatenate([x1, x2, x3, x4, x5])
y = np.concatenate([y1, y2, y3, y4, y5])
z = np.concatenate([z1, z2, z3, z4, z5])

# ------------------------------------------------------------
# VELOCIDADES
# Reproduzindo a lógica usada na Traj original do MATLAB
# ------------------------------------------------------------

vx = np.zeros(N)
vy = np.zeros(N)
vz = np.zeros(N)

vy[0:50] = 0.2 / 50
vx[50:150] = -0.4 / 100
vy[150:250] = -0.4 / 100
vx[250:350] = 0.4 / 100
vy[350:400] = 0.2 / 50

# ------------------------------------------------------------
# ORIENTAÇÃO
# Mesma lógica da Traj original do MATLAB
# ------------------------------------------------------------

theta_z = np.zeros(N)
theta_y = np.zeros(N)
theta_x = np.full(N, np.pi)

theta_z[0:50] = -np.pi + np.arctan(np.linspace(0.0, 0.2, 50) / 0.2)

theta_z[50:100] = -np.pi/2 - np.arctan(np.linspace(0.2, 0.0, 50) / 0.2)
theta_z[100:150] = -np.pi/2 + np.arctan(np.linspace(0.0, 0.2, 50) / 0.2)

theta_z[150:200] = -np.arctan(np.linspace(0.2, 0.0, 50) / 0.2)
theta_z[200:250] = np.arctan(np.linspace(0.0, 0.2, 50) / 0.2)

theta_z[250:300] = np.pi/2 - np.arctan(np.linspace(0.2, 0.0, 50) / 0.2)
theta_z[300:350] = np.pi/2 + np.arctan(np.linspace(0.0, 0.2, 50) / 0.2)

theta_z[350:400] = np.pi - np.arctan(np.linspace(0.2, 0.0, 50) / 0.2)

# ------------------------------------------------------------
# MATRIZ FINAL
# ------------------------------------------------------------
trajetoria = np.column_stack([
    x, y, z, 
    vx, vy, vz, 
    theta_z, theta_y, theta_x
])

# ------------------------------------------------------------
# EXIBIÇÃO
# ------------------------------------------------------------
np.set_printoptions(precision=6, suppress=True)
print("Dimensão da matriz:", trajetoria.shape)
print("\nColunas: [x, y, z, vx, vy, vz, yaw, pitch, roll]")
print("\nPrimeiras 5 linhas:\n", trajetoria[:5])

# ------------------------------------------------------------
# SALVAR EM .MAT
# ------------------------------------------------------------
# Criamos um dicionário onde 'trajetoria' será o nome da variável no MATLAB
data_dict = {"trajetoria": trajetoria}

savemat("trajetoria_quadrado.mat", data_dict)

print("\nArquivo 'trajetoria_quadrado.mat' salvo com sucesso.")
print("Para carregar no MATLAB, use: load('trajetoria_quadrado.mat')")