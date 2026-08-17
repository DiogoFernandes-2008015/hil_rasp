import socket
import time

# Configuração da rede
PI_IP = "0.0.0.0"
PI_PORT = 5006
PC_IP = "192.168.1.50"
PC_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((PI_IP,PI_PORT))
sock.settimeout(1.0)

print("Tentando estabelecer comunicação com o PC")

#Teste de conexão
while True:
	try:
		sock.sendto(b"CONNECT",(PC_IP,PC_PORT)) #envia sinal de presença para o PC
		
		data, addr = sock.recvfrom(1024) #aguarda autorização do PC
		if data.decode() == "START":
			print("Conexão autorizada!")
			break
	except socket.timeout:
		print("Aguardando liberação do usuário no PC")
		time.sleep(1.0
		)

#Parâmetros do controlador
setpoint = 10
Kp = 10
u = 0

sock.settimeout(None) # desativa o timeout de comunicação para a simulação rodar sem interrupções na rede

try:
	while True:
		data, addr = sock.recvfrom(1024)
		msg = data.decode()
		spltmsg = msg.split(";")
		x1 = float(spltmsg[0])
		x2 = float(spltmsg[1])
		x3 = float(spltmsg[2])
		x4 = float(spltmsg[3])
		
		#Controlador 
		u = 22.3607*x1 + 21.0504*x2 - 71.9360*x3 - 13.9142*x4
		print(u)
		sock.sendto(f"{u}".encode(), (PC_IP,PC_PORT))

except KeyboardInterrupt:
	print("Código Interrompido")
finally:
	sock.close()
