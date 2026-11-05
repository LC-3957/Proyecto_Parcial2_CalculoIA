"""
Fase 3: Red neuronal informada por la fisica (PINN) en PyTorch.

Diseno de la estrategia neuronal
--------------------------------
1. Arquitectura: perceptron multicapa (MLP) con activacion tanh (suave e
   infinitamente derivable, necesario porque la perdida usa y'').
2. Restriccion "dura" de condiciones iniciales:
        y_hat(x) = y0 + v0*x + x^2 * N(x/T)
   Con esta forma y_hat(0) = y0 y y_hat'(0) = v0 se cumplen EXACTAMENTE para
   cualquier peso de la red, asi la perdida solo tiene que minimizar el
   residual de la ED (no hay que balancear terminos de perdida).
3. Perdida fisica (no usa ningun dato de la solucion):
        L = mean( (a*y'' + b*y' + c*y - g(x))^2 )   en puntos de colocacion
   Las derivadas y', y'' se obtienen por diferenciacion automatica.
4. Entrenamiento: Adam (exploracion, puntos de colocacion re-muestreados
   aleatoriamente en cada iteracion) seguido de L-BFGS (refinamiento fino
   sobre una malla fija). Precision float64.
5. Seleccion de hiperparametros con el residual en una malla de validacion
   distinta; la solucion exacta NO se usa para elegir el modelo.
"""
import time

import numpy as np
import torch
import torch.nn as nn

from edos import BANCO

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)


def g_torch(ed_id, xt):
    if ed_id == "ED-01":
        return torch.zeros_like(xt)
    if ed_id == "ED-02":
        return 1.0 / torch.cos(3 * xt)
    if ed_id == "ED-03":
        return 2 * torch.exp(xt) + torch.sin(xt)
    raise KeyError(ed_id)


class MLP(nn.Module):
    def __init__(self, ancho=32, capas=3):
        super().__init__()
        dims = [1] + [ancho] * capas + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
        self.red = nn.Sequential(*layers)
        for m in self.red:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t):
        return self.red(t)


class PINN:
    def __init__(self, ed_id, ancho=32, capas=3, semilla=0):
        torch.manual_seed(semilla)
        np.random.seed(semilla)
        self.ed = ed_id
        self.p = BANCO[ed_id]
        self.T = self.p["T"]
        self.red = MLP(ancho, capas)
        self.ancho, self.capas = ancho, capas
        self.historial = []

    def n_parametros(self):
        return sum(q.numel() for q in self.red.parameters())

    def y(self, xt):
        """Ansatz con condiciones iniciales impuestas exactamente."""
        p = self.p
        return p["y0"] + p["v0"] * xt + xt**2 * self.red(xt / self.T)

    def residual(self, xt):
        xt = xt.clone().requires_grad_(True)
        y = self.y(xt)
        dy = torch.autograd.grad(y, xt, torch.ones_like(y), create_graph=True)[0]
        d2y = torch.autograd.grad(dy, xt, torch.ones_like(dy), create_graph=True)[0]
        p = self.p
        return p["a"] * d2y + p["b"] * dy + p["c"] * y - g_torch(self.ed, xt)

    def perdida(self, xt):
        return torch.mean(self.residual(xt) ** 2)

    def entrenar(self, n_colocacion=128, iters_adam=2000, iters_lbfgs=400, lr=1e-3):
        T = self.T
        t0 = time.perf_counter()
        opt = torch.optim.Adam(self.red.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=700, gamma=0.5)
        for it in range(iters_adam):
            xc = torch.rand(n_colocacion, 1) * T  # re-muestreo aleatorio
            opt.zero_grad()
            L = self.perdida(xc)
            L.backward()
            opt.step()
            sched.step()
            if it % 50 == 0:
                self.historial.append(("adam", it, L.item()))

        x_fijo = torch.linspace(0, T, n_colocacion).reshape(-1, 1)
        lbfgs = torch.optim.LBFGS(self.red.parameters(), lr=1.0, max_iter=iters_lbfgs,
                                  tolerance_grad=1e-12, tolerance_change=1e-15,
                                  history_size=50, line_search_fn="strong_wolfe")
        cont = {"k": 0}

        def closure():
            lbfgs.zero_grad()
            L = self.perdida(x_fijo)
            L.backward()
            if cont["k"] % 10 == 0:
                self.historial.append(("lbfgs", iters_adam + cont["k"], L.item()))
            cont["k"] += 1
            return L

        lbfgs.step(closure)
        self.t_entrenamiento = time.perf_counter() - t0
        return self

    def residual_validacion(self, n=1000):
        # malla desplazada: no coincide con los puntos de entrenamiento
        xv = (torch.linspace(0, 1, n) * 0.997 + 0.0013).reshape(-1, 1) * self.T
        return self.perdida(xv).item()

    @torch.no_grad()
    def predecir(self, xs):
        xt = torch.as_tensor(np.asarray(xs, dtype=np.float64)).reshape(-1, 1)
        return self.y(xt).numpy().ravel()

    def guardar(self, ruta):
        torch.save({"estado": self.red.state_dict(), "ed": self.ed,
                    "ancho": self.ancho, "capas": self.capas,
                    "historial": self.historial,
                    "t_entrenamiento": getattr(self, "t_entrenamiento", None)}, ruta)

    @classmethod
    def cargar(cls, ruta):
        d = torch.load(ruta, weights_only=False)
        m = cls(d["ed"], d["ancho"], d["capas"])
        m.red.load_state_dict(d["estado"])
        m.historial = d["historial"]
        m.t_entrenamiento = d["t_entrenamiento"]
        return m
