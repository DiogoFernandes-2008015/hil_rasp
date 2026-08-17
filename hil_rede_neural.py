from __future__ import annotations
import socket
import time
import numpy as np
import base64
import io
from typing import Final
from numpy.typing import ArrayLike, NDArray

INPUT_NAMES: Final[tuple[str, ...]] = (
    "x", "y", "z", "vx", "vy", "vz", "euler_z", "euler_y", "euler_x"
)
OUTPUT_NAMES: Final[tuple[str, ...]] = (
    "theta_1", "theta_2", "theta_3", "theta_4",
    "theta_dot_1", "theta_dot_2", "theta_dot_3", "theta_dot_4",
)

# Pesos, biases e parâmetros de normalização incorporados no próprio arquivo.
_PARAMETROS_BASE64 = """
    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
    CAAAACEAWPN5VP//////////DwAUAHRyYWluX3JhdGlvLm5weQEAEACIAAAAAAAAAEsAAAAAAAAAm+wX6hsQychQxlCtnpJanFyk
    bqWgbpNmoa6joJ6WX1RSlJgXn1+UkgoSd0vMKU4FihdnJBakAvkahjoKhpo6CrUKZAOuNDB4Zg8AUEsDBC0AAAAIAAAAIQA50rXq
    //////////8NABQAdmFsX3JhdGlvLm5weQEAEACIAAAAAAAAAEsAAAAAAAAAm+wX6hsQychQxlCtnpJanFykbqWgbpNmoa6joJ6W
    X1RSlJgXn1+UkgoSd0vMKU4FihdnJBakAvkahjoKhpo6CrUKZAMuYzA4bA8AUEsDBC0AAAAIAAAAIQA50rXq//////////8OABQA
    dGVzdF9yYXRpby5ucHkBABAAiAAAAAAAAABLAAAAAAAAAJvsF+obEMnIUMZQrZ6SWpxcpG6loG6TZqGuo6Cell9UUpSYF59flJIK
    EndLzClOBYoXZyQWpAL5GoY6CoaaOgq1CmQDLmMwOGwPAFBLAwQtAAAACAAAACEAQxQVKv//////////CgAUAGVwb2Nocy5ucHkB
    ABAAiAAAAAAAAABMAAAAAAAAAJvsF+obEMnIUMZQrZ6SWpxcpG6loG6TZqGuo6Cell9UUpSYF59flJIKEndLzClOBYoXZyQWpAL5
    GoY6CoaaOgq1CmQDLgYQODDJAQBQSwMELQAAAAgAAAAhAMp4D7L//////////wwAFABtYXhfZmFpbC5ucHkBABAAiAAAAAAAAABL
    AAAAAAAAAJvsF+obEMnIUMZQrZ6SWpxcpG6loG6TZqGuo6Cell9UUpSYF59flJIKEndLzClOBYoXZyQWpAL5GoY6CoaaOgq1CmQD
    LgYwEHAAAFBLAwQtAAAACAAAACEACysRjv//////////CAAUAGdvYWwubnB5AQAQAIgAAAAAAAAASQAAAAAAAACb7BfqGxDJyFDG
    UK2eklqcXKRupaBuk2ahrqOgnpZfVFKUmBefX5SSChJ3S8wpTgWKF2ckFqQC+RqGOgqGmjoKtQpkAy4GKAAAUEsDBC0AAAAIAAAA
    IQAlzCyD//////////8MABQAbWluX2dyYWQubnB5AQAQAIgAAAAAAAAATwAAAAAAAACb7BfqGxDJyFDGUK2eklqcXKRupaBuk2ah
    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
"""


class RedeNeuralMatlab:
    """Rede feedforward equivalente à NN_On treinada no MATLAB."""

    def __init__(self) -> None:
        raw = base64.b64decode("".join(_PARAMETROS_BASE64.split()))
        with np.load(io.BytesIO(raw), allow_pickle=False) as data:
            for name in (
                "W1", "W2", "W3", "W4", "b1", "b2", "b3", "b4",
                "input_gain", "input_xoffset", "input_ymin",
                "output_gain", "output_xoffset", "output_ymin",
            ):
                setattr(self, name, np.asarray(data[name], dtype=np.float64))
            self.architecture = tuple(int(v) for v in data["architecture"])

        if self.architecture != (9, 16, 16, 16, 8):
            raise ValueError(f"Arquitetura inesperada: {self.architecture}")

    @staticmethod
    def _tansig(x: NDArray[np.float64]) -> NDArray[np.float64]:
        # MATLAB tansig(x) é numericamente equivalente a tanh(x).
        return np.tanh(x)

    @staticmethod
    def _preparar_entrada(values: ArrayLike) -> tuple[NDArray[np.float64], str]:
        x = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(x)):
            raise ValueError("A entrada contém NaN ou infinito.")

        if x.ndim == 1:
            if x.shape[0] != 9:
                raise ValueError(f"A entrada deve ter 9 elementos; recebido {x.shape}.")
            return x.reshape(9, 1), "vetor"

        if x.ndim != 2:
            raise ValueError("Use vetor (9,), matriz (9,N) ou matriz (N,9).")

        if x.shape[0] == 9:
            return x, "variaveis_primeiro"
        if x.shape[1] == 9:
            return x.T, "amostras_primeiro"

        raise ValueError(f"Uma dimensão deve ser 9; recebido {x.shape}.")

    @staticmethod
    def _restaurar_orientacao(y: NDArray[np.float64], orientacao: str) -> NDArray[np.float64]:
        if orientacao == "vetor":
            return y[:, 0]
        if orientacao == "amostras_primeiro":
            return y.T
        return y

    def _forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        # mapminmax.apply do MATLAB
        xp = (x - self.input_xoffset) * self.input_gain + self.input_ymin

        a1 = self._tansig(self.W1 @ xp + self.b1)
        a2 = self._tansig(self.W2 @ a1 + self.b2)
        a3 = self._tansig(self.W3 @ a2 + self.b3)
        yp = self.W4 @ a3 + self.b4  # purelin

        # mapminmax.reverse do MATLAB
        return (yp - self.output_ymin) / self.output_gain + self.output_xoffset

    def predict(self, values: ArrayLike) -> NDArray[np.float64]:
        """Calcula as oito saídas originais da rede."""
        x, orientacao = self._preparar_entrada(values)
        y = self._forward(x)
        return self._restaurar_orientacao(y, orientacao)

    def predict_como_codigo_matlab(self, values: ArrayLike) -> NDArray[np.float64]:
        """Imita o pós-processamento de codigo_unificado_IP.m.

        Mantém os quatro ângulos previstos e substitui as quatro velocidades
        pela diferença entre ângulos consecutivos, sem dividir pelo passo de tempo.
        """
        x, orientacao = self._preparar_entrada(values)
        y = self._forward(x)
        y[4:8, 0] = 0.0
        if y.shape[1] > 1:
            y[4:8, 1:] = np.diff(y[0:4, :], axis=1)
        return self._restaurar_orientacao(y, orientacao)

    __call__ = predict

    
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

# inicialização
dt = 0.001
rede = RedeNeuralMatlab()

sock.settimeout(None)

try:
    while True:
        data, addr = sock.recvfrom(1024)
        msg = data.decode()
        spltmsg = msg.split(";")
        
        xd = float(spltmsg[0])
        yd = float(spltmsg[1])
        zd = float(spltmsg[2])
        vxd = float(spltmsg[3])
        vyd = float(spltmsg[4])
        vzd = float(spltmsg[5])
        yawd = float(spltmsg[6])
        pitchd = float(spltmsg[7])
        rolld = float(spltmsg[8])
        
        entrada = np.array([xd, yd, zd, vxd, vyd, vzd, yawd, pitchd, rolld])
        saida = rede.predict(entrada)
        
        # Correção da F-string para envio de dados
        msg_send = f"{saida[0]};{saida[1]};{saida[2]};{saida[3]};{saida[4]};{saida[5]};{saida[6]};{saida[7]}".encode()
        sock.sendto(msg_send, (PC_IP, PC_PORT))

except KeyboardInterrupt:
    print("\nCódigo Interrompido pelo usuário.")
finally:
    sock.close()
