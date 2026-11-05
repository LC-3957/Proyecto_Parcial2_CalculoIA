"""
Fase 4: Evaluacion comparativa.

Ejecuta los tres paradigmas sobre el banco ED-01..ED-03 y guarda:
  resultados/metricas.csv          MSE, error maximo y costo de cada metodo
  resultados/hiperparametros.csv   barrido de arquitecturas PINN
  resultados/semillas.csv          robustez de la PINN (5 semillas)
  resultados/convergencia_rk4.csv  error de RK4 vs tamano de paso
  resultados/equivalencia.csv      cuantos pasos de RK4 igualan a la PINN
  resultados/extrapolacion.csv     prueba de generalizacion fuera del dominio
  resultados/datos_*.csv           curvas para la interfaz grafica
  modelos/pinn_ED-0X.pt            mejores PINN entrenadas
  figuras/*.png                    graficas del reporte

Uso:  python benchmark.py            (completo, ~10 min en CPU)
      python benchmark.py --rapido   (barrido reducido, ~2 min)
"""
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import analitico
import rk4
from edos import BANCO, exacta_numpy
from pinn import PINN

N_PRUEBA = 501          # puntos de prueba en el dominio
PASOS_RK4 = 500         # h = T/500 (malla de prueba = malla RK4)
RAPIDO = "--rapido" in sys.argv

os.makedirs("resultados", exist_ok=True)
os.makedirs("modelos", exist_ok=True)
os.makedirs("figuras", exist_ok=True)


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def emax(a, b):
    return float(np.max(np.abs(a - b)))


def cronometrar(fn, repeticiones=20):
    """Mediana de varias ejecuciones (reduce el ruido del sistema operativo)."""
    ts = []
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return out, float(np.median(ts))


filas, filas_hp, filas_sem, filas_conv, filas_eq, filas_ext = [], [], [], [], [], []
pasos_analiticos = {}

ANCHOS = [16, 32] if RAPIDO else [16, 32, 64]
CAPAS = [2, 3] if RAPIDO else [2, 3, 4]
SEMILLAS = [0, 1] if RAPIDO else [0, 1, 2, 3, 4]

for ed in BANCO:
    p = BANCO[ed]
    T = p["T"]
    xs = np.linspace(0, T, N_PRUEBA)
    print(f"\n=== {ed}: {p['nombre']}  dominio [0, {T}] ===")

    # ---------------- Analitico ----------------
    t0 = time.perf_counter()
    y_sym, pasos = analitico.resolver(ed)
    t_sym = time.perf_counter() - t0
    ver = analitico.verificar(ed, y_sym)
    assert ver["residual"] == 0 and ver["coincide_con_mano"], ver
    pasos_analiticos[ed] = {"pasos": pasos, "solucion": str(y_sym)}
    f_exacta = exacta_numpy(ed)
    y_ex, t_eval_an = cronometrar(lambda: f_exacta(xs), 200)
    filas.append(dict(ED=ed, metodo="Analítico", MSE=0.0, error_max=0.0,
                      t_preparacion_s=t_sym, t_evaluacion_s=t_eval_an,
                      parametros=0, detalle=p["metodo"]))

    # ---------------- RK4 ----------------
    (x_rk, y_rk), t_rk = cronometrar(lambda: rk4.resolver(ed, PASOS_RK4), 20)
    assert np.allclose(x_rk, xs)
    filas.append(dict(ED=ed, metodo="RK4", MSE=mse(y_rk, y_ex), error_max=emax(y_rk, y_ex),
                      t_preparacion_s=0.0, t_evaluacion_s=t_rk, parametros=0,
                      detalle=f"h = T/{PASOS_RK4} = {T / PASOS_RK4:.4g}"))
    print(f"  RK4  MSE={mse(y_rk, y_ex):.2e}  max={emax(y_rk, y_ex):.2e}  t={t_rk * 1e3:.1f} ms")

    # Convergencia de RK4: error ~ C h^4
    for n in [5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560]:
        (xr, yr), tr = cronometrar(lambda: rk4.resolver(ed, n), 5)
        e = emax(yr, f_exacta(xr))
        filas_conv.append(dict(ED=ed, pasos=n, h=T / n, error_max=e,
                               MSE=mse(yr, f_exacta(xr)), tiempo_s=tr))

    # ---------------- PINN: barrido de hiperparametros ----------------
    mejor, mejor_res = None, np.inf
    for capas, ancho in itertools.product(CAPAS, ANCHOS):
        m = PINN(ed, ancho, capas, semilla=0).entrenar()
        rv = m.residual_validacion()
        yp = m.predecir(xs)
        filas_hp.append(dict(ED=ed, capas=capas, ancho=ancho, parametros=m.n_parametros(),
                             residual_val=rv, MSE=mse(yp, y_ex), error_max=emax(yp, y_ex),
                             t_entrenamiento_s=m.t_entrenamiento))
        print(f"  PINN {capas}x{ancho:<3} res_val={rv:.2e}  MSE={mse(yp, y_ex):.2e}  "
              f"t={m.t_entrenamiento:.1f}s")
        if rv < mejor_res:  # se elige SOLO con el residual (sin usar la exacta)
            mejor, mejor_res = m, rv

    # Robustez: la arquitectura elegida con varias semillas
    for s in SEMILLAS:
        ms = PINN(ed, mejor.ancho, mejor.capas, semilla=s).entrenar()
        yp = ms.predecir(xs)
        filas_sem.append(dict(ED=ed, semilla=s, capas=mejor.capas, ancho=mejor.ancho,
                              MSE=mse(yp, y_ex), error_max=emax(yp, y_ex),
                              t_entrenamiento_s=ms.t_entrenamiento))

    mejor.guardar(f"modelos/pinn_{ed}.pt")
    y_nn, t_inf = cronometrar(lambda: mejor.predecir(xs), 50)
    filas.append(dict(ED=ed, metodo="PINN", MSE=mse(y_nn, y_ex), error_max=emax(y_nn, y_ex),
                      t_preparacion_s=mejor.t_entrenamiento, t_evaluacion_s=t_inf,
                      parametros=mejor.n_parametros(),
                      detalle=f"MLP tanh {mejor.capas} capas x {mejor.ancho} neuronas"))
    print(f"  PINN elegida {mejor.capas}x{mejor.ancho}: MSE={mse(y_nn, y_ex):.2e} "
          f"max={emax(y_nn, y_ex):.2e}")

    # Equivalencia: minimo numero de pasos de RK4 con error max <= el de la PINN
    e_nn = emax(y_nn, y_ex)
    for n in range(2, 5000):
        xr, yr = rk4.resolver(ed, n)
        e_nodos = emax(yr, f_exacta(xr))
        if e_nodos <= e_nn:
            (_, _), tr = cronometrar(lambda: rk4.resolver(ed, n), 50)
            filas_eq.append(dict(ED=ed, error_max_PINN=e_nn, pasos_RK4_equivalentes=n,
                                 error_max_RK4=e_nodos, t_RK4_s=tr,
                                 t_entrenamiento_PINN_s=mejor.t_entrenamiento,
                                 factor_tiempo=mejor.t_entrenamiento / tr))
            break

    # Generalizacion: fuera del dominio de entrenamiento.
    # Ambos metodos se comparan contra la solucion exacta en los nodos de RK4
    # (mismo h que en el dominio), sin interpolar.
    T_ext = 0.52 if ed == "ED-02" else 1.3 * T
    xr, yr = rk4.resolver(ed, int(round(PASOS_RK4 * T_ext / T)), x_fin=T_ext)
    fuera = xr >= T - 1e-12
    xe, yr_e = xr[fuera], yr[fuera]
    ye = f_exacta(xe)
    yn_e = mejor.predecir(xe)
    filas_ext.append(dict(ED=ed, intervalo=f"[{T:g}, {T_ext:g}]",
                          error_max_PINN=emax(yn_e, ye), error_max_RK4=emax(yr_e, ye),
                          error_max_PINN_en_dominio=e_nn))

    # Curvas para la interfaz y las figuras
    xd = np.linspace(0, T_ext, 801)
    pd.DataFrame(dict(x=xs, analitico=y_ex, rk4=y_rk, pinn=y_nn)).to_csv(
        f"resultados/datos_{ed}.csv", index=False)
    pd.DataFrame(dict(x=xd, analitico=f_exacta(xd), pinn=mejor.predecir(xd))).to_csv(
        f"resultados/datos_ext_{ed}.csv", index=False)
    pd.DataFrame(mejor.historial, columns=["fase", "iteracion", "perdida"]).to_csv(
        f"resultados/perdida_{ed}.csv", index=False)

pd.DataFrame(filas).to_csv("resultados/metricas.csv", index=False)
pd.DataFrame(filas_hp).to_csv("resultados/hiperparametros.csv", index=False)
pd.DataFrame(filas_sem).to_csv("resultados/semillas.csv", index=False)
pd.DataFrame(filas_conv).to_csv("resultados/convergencia_rk4.csv", index=False)
pd.DataFrame(filas_eq).to_csv("resultados/equivalencia.csv", index=False)
pd.DataFrame(filas_ext).to_csv("resultados/extrapolacion.csv", index=False)
with open("resultados/analitico.json", "w", encoding="utf-8") as fh:
    json.dump(pasos_analiticos, fh, ensure_ascii=False, indent=2)

print("\nListo. Ejecuta `python figuras.py` para regenerar las graficas.")
