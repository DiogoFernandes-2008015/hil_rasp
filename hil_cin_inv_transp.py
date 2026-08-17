import socket
import time
import numpy as np

def cin_dir(q):
    # q vem como um vetor coluna (3,1), vamos extrair os valores como escalares
    q1, q2, q3 = q[0, 0], q[1, 0], q[2, 0]
    
    a1 = 0.5
    a2 = 0.5
    a3 = 0.5
    sum_q = q1 + q2 + q3
    
    # Adicionado os colchetes externos corretos para o np.array
    A = np.array([
        [np.cos(sum_q), -np.sin(sum_q), 0, a2*np.cos(q1+q2) + a1*np.cos(q1) + a3*np.cos(sum_q)],
        [np.sin(sum_q),  np.cos(sum_q), 0, a2*np.sin(q1+q2) + a1*np.sin(q1) + a3*np.sin(sum_q)],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    xet = np.dot(A, np.array([[0], [0], [0], [1]]))
    xd = np.array([[xet[0, 0]], [xet[1, 0]], [sum_q]])
    return xd
    
def cin_inv(ke, q):
    q1, q2, q3 = q[0, 0], q[1, 0], q[2, 0]
    a1 = 0.5
    a2 = 0.5
    a3 = 0.5
    sum_q = q1 + q2 + q3
    phi2 = 0
    
    J = np.array([
        [-a2*np.sin(q1+q2) - a1*np.sin(q1) - a3*np.sin(sum_q), -a2*np.sin(q1+q2) - a3*np.sin(sum_q), -a3*np.sin(sum_q)],
        [ a2*np.cos(q1+q2) + a1*np.cos(q1) + a3*np.cos(sum_q),  a2*np.cos(q1+q2) + a3*np.cos(sum_q),  a3*np.cos(sum_q)],
        [1, 1, 1]
    ])
    
    TA = np.array([[1, 0, 0], [0, 1, 0], [0, 0, np.cos(phi2)]])
    JA = np.dot(np.linalg.inv(TA), J)
    
    qp = np.dot(np.transpose(JA), ke)
    return qp
    
def integration(q, qp, dt):
    return q + qp * dt
    
# Configuração da rede
PI_IP = "0.0.0.0"
PI_PORT = 5006
PC_IP = "192.168.1.50"
PC_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((PI_IP, PI_PORT))
sock.settimeout(1.0)

print("Tentando estabelecer comunicação com o PC...")

# Teste de conexão
while True:
    try:
        sock.sendto(b"CONNECT", (PC_IP, PC_PORT))
        data, addr = sock.recvfrom(1024)
        if data.decode() == "START":
            print("Conexão autorizada!")
            break
    except socket.timeout:
        print("Aguardando liberação do usuário no PC...")
        time.sleep(1.0)

# Condição inicial e parâmetros auxiliares
PI = np.pi
dt = 0.001
q = np.array([[PI], [-PI/2], [-PI/2]])  # Vetor coluna (3, 1)
K = np.diag([500, 500, 100])

sock.settimeout(None)

try:
    while True:
        data, addr = sock.recvfrom(1024)
        msg = data.decode()
        spltmsg = msg.split(";")
        
        x1 = float(spltmsg[0])
        x2 = float(spltmsg[1])
        x3 = float(spltmsg[2])
        xd = np.array([[x1], [x2], [x3]])
        
        # Cálculo do erro
        e = xd - cin_dir(q)
        
        # Controle P
        ke = np.dot(K, e)
        qp = cin_inv(ke, q)
        q = integration(q, qp, dt)
        
        # Correção da F-string para envio de dados
        msg_send = f"{q[0, 0]};{q[1, 0]};{q[2, 0]}".encode()
        sock.sendto(msg_send, (PC_IP, PC_PORT))

except KeyboardInterrupt:
    print("\nCódigo Interrompido pelo usuário.")
finally:
    sock.close()
