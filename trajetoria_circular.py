import numpy as np
from scipy.io import savemat

# ============================================================
# TRAJETÓRIA CIRCULAR NO PLANO XY
# 
# Matriz final:
# 400 linhas x 9 colunas
#
# Colunas:
# [x, y, z, vx, vy, vz, theta_z, theta_y, theta_x]
# ============================================================

# ------------------------------------------------------------
# PARÂMETROS DA TRAJETÓRIA
# ------------------------------------------------------------
N = 400              # quantidade de pontos
raio = 0.20          # raio do círculo [m]
centro_x = 0.0
centro_y = 0.0

# Criação do vetor angular
phi = np.linspace(0.0, 2.0 * np.pi, N)

# ------------------------------------------------------------
# POSIÇÃO
# ------------------------------------------------------------
x = centro_x + raio * np.cos(phi)
y = centro_y + raio * np.sin(phi)
z = np.zeros(N)

# ------------------------------------------------------------
# VELOCIDADE
# ------------------------------------------------------------
dphi = 2.0 * np.pi / (N - 1)
vx = -raio * dphi * np.sin(phi)
vy =  raio * dphi * np.cos(phi)
vz = np.zeros(N)

# ------------------------------------------------------------
# ORIENTAÇÃO
# ------------------------------------------------------------
theta_z = (phi - np.pi + np.pi) % (2.0 * np.pi) - np.pi
theta_y = np.zeros(N)
theta_x = np.full(N, np.pi)

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

savemat("trajetoria_circular.mat", data_dict)

print("\nArquivo 'trajetoria_circular.mat' salvo com sucesso.")
print("Para carregar no MATLAB, use: load('trajetoria_circular.mat')")