import socket
import time
import numpy as np
import matplotlib.pyplot as plt

def des_traj(t):
    # Retorna um array 1D para simplificar a f-string e a leitura
    x1 = 0.25 - 0.25 * np.cos(np.pi * t)
    x2 = 0.5 + 0.25 * np.sin(np.pi * t)
    phi = np.sin((np.pi / 24) * t)
    return np.array([x1, x2, phi])

def cin_dir(q):
    q1, q2, q3 = q[0, 0], q[1, 0], q[2, 0]
    a1, a2, a3 = 0.5, 0.5, 0.5
    sum_q = q1 + q2 + q3
    
    A = np.array([
        [np.cos(sum_q), -np.sin(sum_q), 0, a2*np.cos(q1+q2) + a1*np.cos(q1) + a3*np.cos(sum_q)],
        [np.sin(sum_q),  np.cos(sum_q), 0, a2*np.sin(q1+q2) + a1*np.sin(q1) + a3*np.sin(sum_q)],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    xet = np.dot(A, np.array([[0], [0], [0], [1]]))
    xd = np.array([[xet[0, 0]], [xet[1, 0]], [sum_q]])
    return xd

# ==========================================
# CONFIGURAÇÕES DE REDE
# ==========================================
PC_IP = "0.0.0.0"       
PC_PORT = 5005
PI_IP = "192.168.1.100"  # IP estático do Raspberry Pi
PI_PORT = 5006

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((PC_IP, PC_PORT))

print("Aguardando sinal de conexão do Raspberry Pi...")

# Handshake inicial
while True:
    data, addr = sock.recvfrom(1024)
    if data.decode() == "CONNECT":
        print(f"Raspberry Pi detectado com sucesso no IP: {addr[0]}")
        break

input("\nComunicação validada! Pressione ENTER para INICIAR a simulação...")

# LIMPEZA DO BUFFER UDP RESIDUAL
sock.setblocking(False)
try:
    while True:
        sock.recvfrom(1024)
except BlockingIOError:
    pass  
sock.setblocking(True)

sock.sendto(b"START", (PI_IP, PI_PORT))

t_zero = time.perf_counter()
t_simulation = 4.0 
dt = 0.001

# Arrays para histórico
xd1_h, xd2_h, xd3_h = [], [], []
q1_h, q2_h, q3_h = [], [], []
xd1hh, xd2hh, xd3hh = [], [], []
times = []  # Corrigido: lista para armazenar o tempo de cada ciclo

sim_time = 0.0  # Corrigido: Inicialização da variável de controle do loop

print("\nSimulação de Alta Performance Iniciada!")

try:
    while sim_time < t_simulation:
        t_start_step = time.perf_counter()
        sim_time = t_start_step - t_zero  # Tempo relativo da simulação
        
        # Gera trajetória com base no tempo de simulação decorrido
        xd = des_traj(sim_time)
        
        xd1_h.append(xd[0])
        xd2_h.append(xd[1])
        xd3_h.append(xd[2])
        
        msg = f"{xd[0]};{xd[1]};{xd[2]}".encode()
        sock.sendto(msg, (PI_IP, PI_PORT))
        
        data, addr = sock.recvfrom(1024)
        msg = data.decode()
        
        if msg == "CONNECT":
            continue  
            
        spltmsg = msg.split(';')
        q1 = float(spltmsg[0])
        q2 = float(spltmsg[1])
        q3 = float(spltmsg[2])
        
        q1_h.append(q1)
        q2_h.append(q2)
        q3_h.append(q3)
        
        # Controle de tempo real
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

    # Processa as posições cartesianas reais calculadas a partir das juntas recebidas
    for i in range(len(q1_h)):
        x_real = cin_dir(np.array([[q1_h[i]], [q2_h[i]], [q3_h[i]]]))
        xd1hh.append(x_real[0, 0])  # Extrai como escalar puro [0, 0]
        xd2hh.append(x_real[1, 0])
        xd3hh.append(x_real[2, 0])

    # Cálculo do erro estático comparando o último ponto real com o último desejado
    if len(xd1_h) > 0 and len(xd1hh) > 0:
        pos_desejada_final = np.array([xd1_h[-1], xd2_h[-1], xd3_h[-1]])
        pos_real_final = np.array([xd1hh[-1], xd2hh[-1], xd3hh[-1]])
        ss_error = np.linalg.norm(pos_desejada_final - pos_real_final)
    else:
        ss_error = 0.0

    print('\n=================== RESULTADOS ===================')
    print(f'Tempo de execução real acumulado: {main_loop_time:.4f} s')
    print(f'Média de duração real de cada passo: {avg_time:.2f} ms')
    print(f'Erro estático final (Norma 2): {ss_error:.4f}')

    # FIGURA 1: Variáveis ao longo do tempo (Passos de simulação)
    plt.figure(figsize=(12, 8))
    labels = ['x1 (Posição X)', 'x2 (Posição Y)', 'x3 (Orientação Phi)']
    dados_desejados = [xd1_h, xd2_h, xd3_h]
    dados_reais = [xd1hh, xd2hh, xd3hh]
    
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(dados_desejados[i], label='Desejado', color='blue', linestyle='--')
        plt.plot(dados_reais[i], label='Real (Pi)', color='red')
        plt.title(f'Comparação para {labels[i]}')
        plt.xlabel('Amostras (Passos)')
        plt.ylabel('Valores')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()

    # =================================================================
    # NOVA FIGURA 2: Trajetória Espacial Espelho (x1 vs x2)
    # =================================================================
    plt.figure(figsize=(7, 7)) # Tamanho quadrado para evitar distorção visual
    plt.plot(xd1_h, xd2_h, label='Trajetória Desejada', color='blue', linestyle='--', linewidth=2)
    plt.plot(xd1hh, xd2hh, label='Trajetória Executada (Pi)', color='red', linewidth=1.5)
    
    # Destaca o ponto de início e fim para facilitar a análise
    if len(xd1_h) > 0:
        plt.scatter(xd1_h[0], xd2_h[0], color='green', marker='o', s=100, label='Início', zorder=5)
        plt.scatter(xd1_h[-1], xd2_h[-1], color='black', marker='X', s=100, label='Fim', zorder=5)

    plt.title('Espaço de Trabalho: Trajetória Cartesiana ($x_1 \times x_2$)')
    plt.xlabel('Posição X ($x_1$) [m]')
    plt.ylabel('Posição Y ($x_2$) [m]')
    plt.axis('equal') # Mantém a proporção 1:1 real dos eixos x e y
    plt.legend()
    plt.grid(True)
    
    # Renderiza ambas as janelas na tela
    plt.show()
    sock.close()