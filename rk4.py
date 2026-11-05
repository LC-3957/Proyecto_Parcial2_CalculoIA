"""Runge-Kutta de orden 4.

La EDO de segundo orden se pasa a un sistema con Y = [y, y'].
"""
import numpy as np

from edos import BANCO, sistema_primer_orden


def rk4(f, x0, Y0, x_fin, n_pasos):
    h = (x_fin - x0) / n_pasos
    xs = np.linspace(x0, x_fin, n_pasos + 1)
    Y = np.zeros((n_pasos + 1, len(Y0)))
    Y[0] = Y0
    for n in range(n_pasos):
        xn, Yn = xs[n], Y[n]
        k1 = f(xn, Yn)
        k2 = f(xn + h / 2, Yn + h / 2 * k1)
        k3 = f(xn + h / 2, Yn + h / 2 * k2)
        k4 = f(xn + h, Yn + h * k3)
        Y[n + 1] = Yn + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return xs, Y


def resolver(ed_id, n_pasos, x_fin=None):
    p = BANCO[ed_id]
    f = sistema_primer_orden(ed_id)
    x_fin = p["T"] if x_fin is None else x_fin
    xs, Y = rk4(f, 0.0, np.array([p["y0"], p["v0"]]), x_fin, n_pasos)
    return xs, Y[:, 0]
