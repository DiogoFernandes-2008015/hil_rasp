# hil_rasp
Python scripts for hardware in the loop simulations.
# 🤖 HIL (Hardware-in-the-Loop) Simulation Framework — PC ↔ Raspberry Pi

A set of Python scripts for **Hardware-in-the-Loop (HIL)** testing of different control and kinematics strategies for robotic manipulators and mobile robots, using real-time communication via **UDP sockets** between a **PC** and a **Raspberry Pi**.

The core idea is to split the simulation into two ends that exchange data at every time step:

- **PC**: typically responsible for generating the reference trajectory/simulated plant (or environment), measuring latency, plotting results and exporting data.
- **Raspberry Pi**: typically responsible for running the embedded controller/algorithm in real time (the "hardware" in the loop).

Each pair of files (`*_pc.py` on the PC and its counterpart without the suffix on the Pi, or vice versa) implements a different control scenario, allowing performance, network latency and closed-loop behavior to be compared.

---

## 🧠 Implemented Control Scenarios

| Scenario | PC script | Pi script | Description |
|---|---|---|---|
| **PID — 1st-order system** | `hilpid.py` | `hilrasp.py` | Proportional control of a 1st-order plant simulated on the PC; the Pi computes the control action `u` from the error. Measures RTT (latency) on every cycle. |
| **State feedback / Inverted pendulum** | `hil_pend_inv.py` | `hilraspmltvar.py` | The PC simulates an inverted pendulum (4-state state-space model) and sends the state vector to the Pi; the Pi computes `u` via a fixed-gain state-feedback control law and sends it back to the PC, which integrates the dynamics. |
| **Inverse kinematics — Jacobian inverse** | `hil_cin_inv_inv_pc.py` | `hil_cin_inv_inv.py` | The PC generates a desired Cartesian trajectory (position + velocity feedforward) for a 3-link planar manipulator; the Pi solves the inverse kinematics via the Jacobian inverse and integrates the joint positions. Generates plots comparing the desired vs. executed trajectory. |
| **Inverse kinematics — Jacobian transpose** | `hil_cin_inv_transp_pc.py` | `hil_cin_inv_transp.py` | Same structure as the previous scenario, but the inverse kinematics is solved via the **Jacobian transpose** (no velocity feedforward). |
| **Neural-network-based control** | `hil_rede_neural_pc.py` | `hil_rede_neural.py` | The PC loads a pre-generated trajectory (circular or square, `.mat`) and streams it point by point to the Pi; the Pi runs a neural-network-based controller/estimator to compute the joint values `q1..q4` (and their derivatives), which are sent back to the PC and exported to `.mat`. |
| **MPC (Model Predictive Control) with CasADi** | *(plant/environment not included)* | `hil_mpc_casadi.py` | On every cycle, the Pi solves a nonlinear optimization problem (via CasADi + IPOPT) to drive an omnidirectional (4-wheel) mobile robot to a target state `(x, y, θ)`, with prediction horizon `N` and RK4 integration. |

> 📌 In the inverse-kinematics and neural-network scenarios, the actual **controller** runs **on the Pi**; the reference trajectory/plant generation, plotting and result export run **on the PC** — even if a filename suggests otherwise at first glance, follow the columns in the table above.

---

## 🔌 Communication Protocol (UDP handshake)

All scenarios follow the same synchronization pattern before starting the real-time control loop:

1. The **Pi** repeatedly sends the `b"CONNECT"` packet to the PC until it gets a reply.
2. The **PC** detects the `CONNECT`, prints the source IP, and waits for manual user confirmation (`ENTER`) to start.
3. The **PC** sends `b"START"` to the Pi, releasing the main loop.
4. Closed-loop control loop, exchanging text messages separated by `;` (e.g., `"x1;x2;x3"`).
5. At the end, the PC (or the Pi, depending on the script) sends `b"STOP"` to cleanly terminate the simulation.

**Network configuration:** every script defines `PC_IP`, `PC_PORT`, `PI_IP` and `PI_PORT` at the top of the file — update these to the actual IPs on your network before running (the IPs in the code are just placeholders/examples).

---

## 🛤️ Trajectory Generation

| Script | Description |
|---|---|
| `trajetoria_circular.py` | Generates a circular trajectory (0.20 m radius, 400 points) in the XY plane, with position, velocity and orientation, saved to `trajetoria_circular.mat`. |
| `trajetoria_quadrado.py` | Generates a square-shaped trajectory (0.40 m side, 400 points), with position, velocity and orientation, saved to `trajetoria_quadrado.mat`. |

Both scripts print the shape of the generated matrix (400×9) and the first rows for verification, with columns `[x, y, z, vx, vy, vz, yaw, pitch, roll]`.

---

## 📁 Data Files (`.mat` / `.csv`)

| File | Source | Content |
|---|---|---|
| `trajetoria_circular.mat` / `.csv` | `trajetoria_circular.py` | Circular reference trajectory (simulation input) |
| `trajetoria_quadrado.mat` | `trajetoria_quadrado.py` | Square reference trajectory (simulation input) |
| `resultados_simulacao_circular.mat` | `hil_rede_neural_pc.py` | HIL simulation results for the circular trajectory (desired positions, joint values `q1..q4`, joint velocities, and execution times) |
| `resultados_simulacao_quadrado.mat` | `hil_rede_neural_pc.py` | HIL simulation results for the square trajectory (same variables as above) |

> To switch between the circular and square trajectory in the neural-network scenario, change the `.mat` file loaded in `hil_rede_neural_pc.py` (`loadmat("trajetoria_quadrado.mat")` → `loadmat("trajetoria_circular.mat")`) and the output filename (`resultados_simulacao_quadrado.mat` → `resultados_simulacao_circular.mat`).

---

## 📦 Requirements

- Python 3
- Python libraries:
  - `numpy`
  - `scipy`
  - `pandas`
  - `matplotlib`
  - `casadi` (only for the MPC scenario)

### Installation

```bash
pip3 install numpy scipy pandas matplotlib casadi
```

> On the Raspberry Pi, it's recommended to use pre-built `numpy`/`scipy` wheels (via `apt` or `piwheels`) to avoid long compilation times.

---

## 🚀 How to Run

Each scenario requires both scripts to run simultaneously — **start on the Pi first, then on the PC** (the Pi keeps trying to connect until the PC responds):

1. **On the Raspberry Pi**, run the script for the desired scenario, e.g.:
   ```bash
   python3 hilrasp.py
   ```

2. **On the PC**, run the corresponding script:
   ```bash
   python3 hilpid.py
   ```

3. Wait for the PC terminal to detect the Pi, then press **ENTER** to start the simulation.

4. For scenarios that require a trajectory (`trajetoria_circular.py` / `trajetoria_quadrado.py`), run them **first** to generate the input `.mat` files:
   ```bash
   python3 trajetoria_circular.py
   python3 trajetoria_quadrado.py
   ```

---

## ⚠️ Notes & Limitations

- Network IPs (`PC_IP`, `PI_IP`) are hardcoded as examples — update them to match your local network (a static IP on both devices is recommended).
- Communication is done over **UDP with no delivery guarantee**; on unstable networks, packets may be lost, which can cause the main loops to hang on `recvfrom()`. For field testing, consider adding timeouts/retries to the main loops.
- `hil_mpc_casadi.py` relies on `IPOPT` (via CasADi) to solve the optimization problem in real time — on limited hardware (like the Raspberry Pi), solve time may be the limiting factor for the control frequency.
- The `hil_rede_neural.py` script (Raspberry Pi side) is not included in this file set — only its counterpart `hil_rede_neural_pc.py` is.
- `hil_pend_inv.py` references an undefined variable (`Ax3_h`) in the final plotting block (`except KeyboardInterrupt`); review this line before running it to completion, as it should likely refer to `x3_h`.

---

## ✍️ Author

Developed by **Diogo Lopes Fernandes**.
