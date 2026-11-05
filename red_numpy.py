"""Evalúa la PINN ya entrenada con NumPy para que la app no necesite PyTorch.

Los pesos se generan con exportar_pesos.py.
"""
import numpy as np

from edos import BANCO


class RedNumpy:
    def __init__(self, ed_id, pesos, sesgos, capas, ancho, t_entrenamiento):
        self.ed = ed_id
        self.p = BANCO[ed_id]
        self.W, self.b = pesos, sesgos
        self.capas, self.ancho = capas, ancho
        self.t_entrenamiento = t_entrenamiento

    @classmethod
    def cargar(cls, ruta, ed_id):
        d = np.load(ruta)
        n = int(d["n_lineales"])
        return cls(ed_id, [d[f"W{i}"] for i in range(n)], [d[f"b{i}"] for i in range(n)],
                   int(d["capas"]), int(d["ancho"]), float(d["t_entrenamiento"]))

    def n_parametros(self):
        return int(sum(w.size for w in self.W) + sum(b.size for b in self.b))

    def red(self, t):
        h = t
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W.T + b
            if i < len(self.W) - 1:
                h = np.tanh(h)
        return h

    def predecir(self, xs):
        x = np.asarray(xs, dtype=np.float64).reshape(-1, 1)
        p = self.p
        return (p["y0"] + p["v0"] * x + x**2 * self.red(x / p["T"])).ravel()
