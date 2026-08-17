import socket
import time
import numpy as np
import matplotlib.pyplot as plt

# Configurações de Rede
PC_IP = "0.0.0.0"       # Ouve em todas as interfaces de rede
PC_PORT = 5005
PI_IP = "192.168.1.100"  # IP estático do seu Raspberry Pi
PI_PORT = 5006

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((PC_IP, PC_PORT))

#Variáveis para armazenamento do histórico
tempo_h = []
x3_h = []
x1_h = []
u_h = []


print("Aguardando sinal de conexão do Raspberry Pi...")

# 1. TESTE DE COMUNICAÇÃO (Handshake)
while True:
    data, addr = sock.recvfrom(1024)
    if data.decode() == "CONNECT":
        print(f"Raspberry Pi detectado com sucesso no IP: {addr[0]}")
        break

# REQUISITO 1: Confirmação do usuário no PC
input("\nComunicação validada! Pressione ENTER para INICIAR a simulação...")

sock.sendto(b"START", (PI_IP, PI_PORT))
 
t_zero = time.perf_counter()

x = np.array([[0],[0],[3.14],[0]])        # Estado inicial
dt = 0.001      # Passo de tempo da simulação (20ms)

print("\n Simulação iniciada em malha fechada!")

# Envia o estado inicial para dar a partida no loop do Pi
x0 = x[0].item() 
x1 = x[1].item()
x2 = x[2].item()
x3 = x[3].item()

#Cria uma string para ser enviada
msg = f"{x0};{x1};{x2};{x3}".encode()

# Envia a mensagem atualizado para o Raspberry Pi
sock.sendto(msg, (PI_IP, PI_PORT))


try:
    while True:
        # REQUISITO 2: Medição de tempo (Início do envio/espera)
        t_envio = time.perf_counter()

        # Espera receber a ação de controle (u) do Pi
        data, addr = sock.recvfrom(1024)
        
        try:
            u = float(data.decode())
        except ValueError:
            continue
        
        # REQUISITO 2: Calcula o tempo decorrido até a resposta voltar (RTT)
        latencia_ms = (time.perf_counter() - t_envio) * 1000

        # Atualiza a física do modelo matemático
        A = np.array([[0,1,0,0],[0,-0.1818,2.6754,0],[0,0,0,1],[0,-0.4545,31.2136,0]])
        B = np.array([[0],[1.8182],[0],[4.5454]])
        xp = A @ x + B * u
        x = x + xp * dt
        x0 = x[0].item() 
        x1 = x[1].item()
        x2 = x[2].item()
        x3 = x[3].item()
        
        
        tempo_h.append(time.perf_counter()-t_zero)
        x1_h.append(x1)
        x3_h.append(x3)
        u_h.append(u)
        #Cria uma string para ser enviada
        msg = f"{x0};{x1};{x2};{x3}".encode()
        
        # Envia a mensagem atualizado para o Raspberry Pi
        sock.sendto(msg, (PI_IP, PI_PORT))

        # Telemetria no terminal
        print(f"Latência (RTT): {latencia_ms:.2f} ms | U: {u:.6f}| Theta = {x[2].item()}" )

except KeyboardInterrupt:
    print("\nSimulação finalizada pelo usuário.")
    
    plt.figure(figsize = (12,8))
    
    plt.subplot(2,1,1)
    plt.plot(tempo_h,x1_h,label='x1')
    plt.plot(tempo_h,Ax3_h,label='x3')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2,1,2)
    plt.plot(tempo_h,u_h)
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
finally:
    sock.close()