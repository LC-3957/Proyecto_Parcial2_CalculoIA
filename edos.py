"""
Banco de ecuaciones diferenciales de prueba.

Todas son lineales de segundo orden con coeficientes constantes:
    a*y'' + b*y' + c*y = g(x),   y(0) = y0,  y'(0) = v0

Cada ED se resuelve en un dominio [0, T]. Para ED-02 el dominio se limita a
[0, 0.5] porque sec(3x) tiene una asintota en x = pi/6 ~ 0.5236 y la solucion
solo existe en el intervalo (-pi/6, pi/6).
"""
import numpy as np
import sympy as sp

x = sp.symbols("x", real=True)

BANCO = {
    "ED-01": dict(
        nombre="y'' + 4y' + 4y = 0",
        a=1, b=4, c=4,
        g=sp.Integer(0),
        y0=1.0, v0=0.0,
        T=5.0,
        metodo="Coeficientes constantes",
    ),
    "ED-02": dict(
        nombre="y'' + 9y = sec(3x)",
        a=1, b=0, c=9,
        g=sp.sec(3 * x),
        y0=0.0, v0=0.0,
        T=0.5,
        metodo="Variación de parámetros",
    ),
    "ED-03": dict(
        nombre="y'' - 5y' + 6y = 2e^x + sin(x)",
        a=1, b=-5, c=6,
        g=2 * sp.exp(x) + sp.sin(x),
        y0=1.0, v0=2.0,
        T=1.0,
        metodo="Superposición",
    ),
}

# Soluciones exactas cerradas obtenidas a mano (ver reporte). Se usan como
# referencia "verdadera" y para validar que el solucionador simbolico
# programado en analitico.py llega exactamente al mismo resultado.
EXACTAS = {
    "ED-01": (1 + 2 * x) * sp.exp(-2 * x),
    "ED-02": sp.Rational(1, 9) * sp.cos(3 * x) * sp.log(sp.cos(3 * x))
             + x / 3 * sp.sin(3 * x),
    "ED-03": -sp.Rational(6, 5) * sp.exp(2 * x) + sp.Rational(11, 10) * sp.exp(3 * x)
             + sp.exp(x) + (sp.cos(x) + sp.sin(x)) / 10,
}


def g_numpy(ed_id):
    """Funcion de forzamiento g(x) vectorizada con NumPy."""
    return sp.lambdify(x, BANCO[ed_id]["g"] + 0 * x, "numpy")


def exacta_numpy(ed_id):
    return sp.lambdify(x, EXACTAS[ed_id], "numpy")


def sistema_primer_orden(ed_id):
    """Convierte la ED de 2o orden en un sistema y1' = y2, y2' = (g - b y2 - c y1)/a."""
    p = BANCO[ed_id]
    a, b, c = p["a"], p["b"], p["c"]
    g = g_numpy(ed_id)

    def f(xv, Y):
        y1, y2 = Y
        return np.array([y2, (g(xv) - b * y2 - c * y1) / a])

    return f
