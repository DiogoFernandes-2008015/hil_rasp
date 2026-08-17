import socket
import time
import numpy as np

# Configurações de Rede
PC_IP = "0.0.0.0"       # Ouve em todas as interfaces de rede
PC_PORT = 5005
PI_IP = "192.168.1.100"  # IP estático do seu Raspberry Pi
PI_PORT = 5006

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((PC_IP, PC_PORT))


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

# Parâmetros da Planta Física (Exemplo: Sistema de 1ª ordem)
# dx/dt = -a*x + b*u
a, b = 0.5, 2.0
x = 0.0        # Estado inicial (ex: velocidade)
dt = 0.02      # Passo de tempo da simulação (20ms)

print("\n Simulação iniciada em malha fechada!")

# Envia o estado inicial para dar a partida no loop do Pi
sock.sendto(f"{x}".encode(), (PI_IP, PI_PORT))

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

        # Atualiza a física do modelo matemático (Integração de Euler)
        dxdt = -a * x + b * u
        x = x + dxdt * dt
        x = max(0.0, x)  # Saturação física mínima (ex: velocidade não cai abaixo de 0)

        # Envia o novo estado atualizado para o Raspberry Pi
        sock.sendto(f"{x}".encode(), (PI_IP, PI_PORT))

        # Telemetria no terminal
        print(f"Latência (RTT): {latencia_ms:.2f} ms | U: {u:.2f} | Estado X: {x:.2f}")

except KeyboardInterrupt:
    print("\nSimulação finalizada pelo usuário.")
finally:
    sock.close()