import socket
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat, savemat  # <-- Adicionado savemat para exportar os dados

# ==========================================
# IMPORTAÇÃO DOS DADOS DE TRAJETÓRIA (.MAT)
# ==========================================
dados_mat = loadmat("trajetoria_quadrado.mat")
trajetoria_array = dados_mat["trajetoria"]  # Extrai a matriz (400x9)

df_trajetoria = pd.DataFrame(trajetoria_array, columns=[
    'x', 'y', 'z', 'vx', 'vy', 'vz', 'yaw', 'pitch', 'roll'
])

total_passos = len(df_trajetoria)

# ==========================================
# CONFIGURAÇÕES DE REDE
# ==========================================
PC_IP = "0.0.0.0"       
PC_PORT = 5005
PI_IP = "192.168.1.100"  
PI_PORT = 5006

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((PC_IP, PC_PORT))

print("Aguardando sinal de conexão do Raspberry Pi...")
while True:
    data, addr = sock.recvfrom(1024)
    if data.decode() == "CONNECT":
        print(f"Raspberry Pi detectado com sucesso no IP: {addr[0]}")
        break

input("\nComunicação validada! Pressione ENTER para INICIAR a simulação...")

sock.setblocking(False)
try:
    while True:
        sock.recvfrom(1024)
except BlockingIOError:
    pass  

sock.setblocking(True)
sock.sendto(b"START", (PI_IP, PI_PORT))

t_zero = time.perf_counter()
dt = 0.001

# Arrays para histórico
q1_h, q2_h, q3_h, q4_h = [], [], [], []
q1p_h, q2p_h, q3p_h, q4p_h = [], [], [], []
xd_h, yd_h, zd_h = [], [], []
vxd_h, vyd_h, vzd_h = [], [], []
yawd_h, pitchd_h, rolld_h = [], [], []
times = []

step = 0  # Índice para percorrer a tabela de dados

print("\nSimulação de Alta Performance Iniciada!")

try:
    while step < total_passos:
        t_start_step = time.perf_counter()
        
        ponto_atual = df_trajetoria.iloc[step]
        
        xd     = float(ponto_atual['x'])
        yd     = float(ponto_atual['y'])
        zd     = float(ponto_atual['z'])
        vxd    = float(ponto_atual['vx'])
        vyd    = float(ponto_atual['vy'])
        vzd    = float(ponto_atual['vz'])
        yawd   = float(ponto_atual['yaw'])
        pitchd = float(ponto_atual['pitch'])
        rolld  = float(ponto_atual['roll'])
        
        xd_h.append(xd)
        yd_h.append(yd)
        zd_h.append(zd)
        vxd_h.append(vxd)
        vyd_h.append(vyd)
        vzd_h.append(vzd)
        yawd_h.append(yawd)
        pitchd_h.append(pitchd)
        rolld_h.append(rolld)

        msg = f"{xd};{yd};{zd};{vxd};{vyd};{vzd};{yawd};{pitchd};{rolld}".encode()
        sock.sendto(msg, (PI_IP, PI_PORT))
        
        data, addr = sock.recvfrom(1024)
        msg = data.decode()
        
        if msg == "CONNECT":
            continue  
            
        spltmsg = msg.split(';')
        q1, q2, q3, q4 = map(float, spltmsg[0:4])
        q1p, q2p, q3p, q4p = map(float, spltmsg[4:8])
        
        q1_h.append(q1); q2_h.append(q2); q3_h.append(q3); q4_h.append(q4)
        q1p_h.append(q1p); q2p_h.append(q2p); q3p_h.append(q3p); q4p_h.append(q4p)
        
        step += 1  
        
        t_computacao_passo = time.perf_counter() - t_start_step
        times.append(t_computacao_passo)
        
        if t_computacao_passo < dt:
            time.sleep(dt - t_computacao_passo)
            
    sock.sendto(b"STOP", (PI_IP, PI_PORT))

except KeyboardInterrupt:
    print("\nSimulação finalizada pelo usuário via teclado.")
    try:
        sock.sendto(b"STOP", (PI_IP, PI_PORT))
    except:
        pass

finally:
    main_loop_time = time.perf_counter() - t_zero
    avg_time = np.array(times).mean() * 1000 if len(times) > 0 else 0.0

    print('\n=================== RESULTADOS ===================')
    print(f'Tempo de execução real acumulado: {main_loop_time:.4f} s')
    print(f'Média de duração real de cada passo: {avg_time:.2f} ms')
    print(f'Pontos processados: {step}/{total_passos}')

    # ==========================================
    # EXPORTAÇÃO DOS DADOS LIDOS PARA .MAT
    # ==========================================
    dados_saida = {
        "xd": np.array(xd_h),
        "yd": np.array(yd_h),
        "zd": np.array(zd_h),
        "vxd": np.array(vxd_h),
        "vyd": np.array(vyd_h),
        "vzd": np.array(vzd_h),
        "yawd": np.array(yawd_h),
        "pitchd": np.array(pitchd_h),
        "rolld": np.array(rolld_h),
        "q1": np.array(q1_h),
        "q2": np.array(q2_h),
        "q3": np.array(q3_h),
        "q4": np.array(q4_h),
        "q1p": np.array(q1p_h),
        "q2p": np.array(q2p_h),
        "q3p": np.array(q3p_h),
        "q4p": np.array(q4p_h),
        "times": np.array(times)
    }

    savemat("resultados_simulacao_quadrado.mat", dados_saida)
    print("Dados da simulação exportados com sucesso para 'resultados_simulacao_quadrado.mat'!")
        
    sock.close()