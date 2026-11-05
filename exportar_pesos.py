"""Guarda los pesos de modelos/*.pt como .npz para red_numpy.py."""
import numpy as np
import torch.nn as nn

from edos import BANCO
from pinn import PINN

for ed in BANCO:
    m = PINN.cargar(f"modelos/pinn_{ed}.pt")
    lin = [c for c in m.red.red if isinstance(c, nn.Linear)]
    datos = {f"W{i}": c.weight.detach().numpy() for i, c in enumerate(lin)}
    datos.update({f"b{i}": c.bias.detach().numpy() for i, c in enumerate(lin)})
    np.savez(f"modelos/pinn_{ed}.npz", n_lineales=len(lin), capas=m.capas, ancho=m.ancho,
             t_entrenamiento=m.t_entrenamiento, **datos)
    print(ed, "exportado")
