import socket
import casadi as ca
import numpy as np
from casadi import sin, cos, pi

# ==========================================
# CONFIGURAÇÕES DE REDE
# ==========================================
PC_IP = "192.168.1.50"   # <--- COLOQUE O IP REAL DA PLACA ETHERNET DO SEU PC AQUI!
PC_PORT = 5005
PI_IP = "0.0.0.0"        
PI_PORT = 5006

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((PI_IP, PI_PORT))

# ==========================================
# PARÂMETROS E CONFIGURAÇÃO DO CASADI (MPC)
# ==========================================
Q_x, Q_y, Q_theta = 100, 100, 2000
R1, R2, R3, R4 = 1, 1, 1, 1

step_horizon = 0.1 
N = 5             
wheel_radius = 1   
Lx, Ly = 0.3, 0.3           
v_max, v_min = 1, -1

x_target, y_target, theta_target = 15, 10, pi/4
state_target = ca.DM([x_target, y_target, theta_target])

x, y, theta = ca.SX.sym('x'), ca.SX.sym('y'), ca.SX.sym('theta')
states = ca.vertcat(x, y, theta)
n_states = states.numel()

V_a, V_b, V_c, V_d = ca.SX.sym('V_a'), ca.SX.sym('V_b'), ca.SX.sym('V_c'), ca.SX.sym('V_d')
controls = ca.vertcat(V_a, V_b, V_c, V_d)
n_controls = controls.numel()

X = ca.SX.sym('X', n_states, N + 1)
U = ca.SX.sym('U', n_controls, N)
P = ca.SX.sym('P', n_states + n_states)

Q_mat = ca.diagcat(Q_x, Q_y, Q_theta)
R_mat = ca.diagcat(R1, R2, R3, R4)

rot_3d_z = ca.vertcat(
    ca.horzcat(cos(theta), -sin(theta), 0),
    ca.horzcat(sin(theta),  cos(theta), 0),
    ca.horzcat(         0,           0, 1)
)
J = (wheel_radius/4) * ca.DM([
    [         1,         1,          1,         1],
    [        -1,         1,          1,        -1],
    [-1/(Lx+Ly), 1/(Lx+Ly), -1/(Lx+Ly), 1/(Lx+Ly)]
])
RHS = rot_3d_z @ J @ controls
f = ca.Function('f', [states, controls], [RHS])

cost_fn = 0 
g = X[:, 0] - P[:n_states] 

for k in range(N):
    st = X[:, k]
    con = U[:, k]
    cost_fn = cost_fn + (st - P[n_states:]).T @ Q_mat @ (st - P[n_states:]) + con.T @ R_mat @ con
    st_next = X[:, k+1]
    k1 = f(st, con)
    k2 = f(st + step_horizon/2*k1, con)
    k3 = f(st + step_horizon/2*k2, con)
    k4 = f(st + step_horizon * k3, con)
    st_next_RK4 = st + (step_horizon / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    g = ca.vertcat(g, st_next - st_next_RK4)

OPT_variables = ca.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1)))
nlp_prob = {'f': cost_fn, 'x': OPT_variables, 'g': g, 'p': P}

opts = {
    'ipopt': {
        'max_iter': 2000,
        'print_level': 0,
        'acceptable_tol': 1e-8,
        'acceptable_obj_change_tol': 1e-6
    },
    'print_time': 0
}
solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

lbx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))
ubx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))
lbx[0: n_states*(N+1): n_states] = -ca.inf    
lbx[1: n_states*(N+1): n_states] = -ca.inf    
lbx[2: n_states*(N+1): n_states] = -ca.inf    
ubx[0: n_states*(N+1): n_states] = ca.inf     
ubx[1: n_states*(N+1): n_states] = ca.inf     
ubx[2: n_states*(N+1): n_states] = ca.inf     
lbx[n_states*(N+1):] = v_min                 
ubx[n_states*(N+1):] = v_max                 

args = {
    'lbg': ca.DM.zeros((n_states*(N+1), 1)), 
    'ubg': ca.DM.zeros((n_states*(N+1), 1)), 
    'lbx': lbx,
    'ubx': ubx
}

u0 = ca.DM.zeros((n_controls, N)) 
X0 = ca.DM.zeros((n_states, N+1))

print("Enviando sinal de conexão para o PC...")
while True:
    sock.sendto(b"CONNECT", (PC_IP, PC_PORT))
    sock.settimeout(1.0)
    try:
        data, addr = sock.recvfrom(1024)
        if data.decode() == "START":
            print("Conexão efetuada com sucesso! Inicializando Malha Fechada...")
            break
    except socket.timeout:
        continue

sock.settimeout(None) 
first_iter = True

# Loop de Controle Otimizado
while True:
    data, addr = sock.recvfrom(1024)
    msg = data.decode()
    
    if msg == "STOP":
        print("Sinal de parada enviado pelo PC recebido. Encerrando o programa...")
        break
        
    try:
        x_val, y_val, theta_val = map(float, msg.split(';'))
        state_init = ca.DM([x_val, y_val, theta_val])
    except ValueError:
        continue

    if first_iter:
        X0 = ca.repmat(state_init, 1, N+1)
        first_iter = False

    args['p'] = ca.vertcat(state_init, state_target)
    args['x0'] = ca.vertcat(
        ca.reshape(X0, n_states*(N+1), 1),
        ca.reshape(u0, n_controls*N, 1)
    )

    sol = solver(
        x0=args['x0'], lbx=args['lbx'], ubx=args['ubx'],
        lbg=args['lbg'], ubg=args['ubg'], p=args['p']
    )

    u_sol = ca.reshape(sol['x'][n_states * (N + 1):], n_controls, N)
    X0_sol = ca.reshape(sol['x'][: n_states * (N+1)], n_states, N+1)

    # OTIMIZAÇÃO EXTRA: Extrai apenas a ação do instante atual u[:, 0] e ignora os estados X0_sol no envio
    u_next = u_sol[:, 0].full().flatten()
    u_str = ",".join(map(str, u_next))
    
    # Envia de volta estritamente as entradas necessárias para os motores
    sock.sendto(u_str.encode(), addr)

    # O Pi continua mantendo os estados calculados internamente para a lógica de warm start
    u0 = ca.horzcat(u_sol[:, 1:], ca.reshape(u_sol[:, -1], -1, 1))
    X0 = ca.horzcat(X0_sol[:, 1:], ca.reshape(X0_sol[:, -1], -1, 1))

sock.close()
