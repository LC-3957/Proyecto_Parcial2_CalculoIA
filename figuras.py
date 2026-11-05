"""Genera las graficas del reporte a partir de resultados/*.csv."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from edos import BANCO

os.makedirs("figuras", exist_ok=True)

COL = {"Analítico": "#2a78d6", "RK4": "#eb6834", "PINN": "#1baf7a"}
TXT, TXT2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TXT2,
    "xtick.color": TXT2, "ytick.color": TXT2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlecolor": TXT,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
})
EDS = list(BANCO)


def guardar(fig, nombre):
    fig.savefig(f"figuras/{nombre}.png")
    plt.close(fig)


# 1. Soluciones superpuestas
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, ed in zip(axs, EDS):
    d = pd.read_csv(f"resultados/datos_{ed}.csv")
    ax.plot(d.x, d.analitico, color=COL["Analítico"], lw=2.4, label="Analítico (exacta)")
    ax.plot(d.x, d.rk4, color=COL["RK4"], lw=2, ls="--", label="RK4")
    ax.plot(d.x[::25], d.pinn[::25], "o", ms=5, mfc="white", mec=COL["PINN"], mew=1.6,
            label="PINN")
    ax.set_title(f"{ed}: {BANCO[ed]['nombre']}", fontsize=10)
    ax.set_xlabel("x"); ax.set_ylabel("y(x)")
axs[0].legend(loc="upper right")
fig.suptitle("Las tres soluciones son visualmente indistinguibles", x=0.01, y=1.06, ha="left",
             fontweight="bold", color=TXT)
guardar(fig, "fig1_soluciones")

# 2. Error puntual (escala log)
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for ax, ed in zip(axs, EDS):
    d = pd.read_csv(f"resultados/datos_{ed}.csv")
    ax.semilogy(d.x, np.abs(d.rk4 - d.analitico) + 1e-18, color=COL["RK4"], lw=2, label="RK4")
    ax.semilogy(d.x, np.abs(d.pinn - d.analitico) + 1e-18, color=COL["PINN"], lw=2, label="PINN")
    ax.set_title(ed); ax.set_xlabel("x")
axs[0].set_ylabel("|error| respecto a la exacta")
axs[0].legend(loc="lower right")
fig.suptitle("Error absoluto punto a punto: RK4 (h=T/500) es varios órdenes más preciso",
             x=0.01, y=1.06, ha="left", fontweight="bold", color=TXT)
guardar(fig, "fig2_error_puntual")

# 3. Convergencia RK4
c = pd.read_csv("resultados/convergencia_rk4.csv")
fig, ax = plt.subplots(figsize=(6.4, 4.2))
marcas = {"ED-01": "o", "ED-02": "s", "ED-03": "^"}
tonos = {"ED-01": "#6250d6", "ED-02": "#c0447a", "ED-03": "#8a6a00"}
for ed in EDS:
    s = c[c.ED == ed]
    ax.loglog(s.pasos, s.error_max, marker=marcas[ed], color=tonos[ed], lw=2, ms=6, label=ed)
ref_n = np.array([20, 1280])
ax.loglog(ref_n, 2e-1 * (ref_n / 20.0) ** -4, color=TXT2, ls=":", lw=1.5)
ax.text(160, 2e-1 * (160 / 20) ** -4 * 3, "pendiente −4 (O(h⁴))", color=TXT2)
ax.set_xlabel("número de pasos (n = T/h)"); ax.set_ylabel("error absoluto máximo")
ax.legend()
ax.set_title("Convergencia de RK4: el error cae ~16× al duplicar los pasos", loc="left")
guardar(fig, "fig3_convergencia_rk4")

# 4. Historial de perdida PINN
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for ed in EDS:
    h = pd.read_csv(f"resultados/perdida_{ed}.csv")
    ax.semilogy(h.iteracion, h.perdida, color=tonos[ed], lw=1.8, label=ed)
corte = pd.read_csv(f"resultados/perdida_{EDS[0]}.csv").query("fase=='lbfgs'").iteracion.min()
ax.axvline(corte, color=TXT2, lw=1, ls="--")
ax.text(corte * 0.97, ax.get_ylim()[1] * 0.3, "Adam → L-BFGS", ha="right", color=TXT2)
ax.set_xlabel("iteración"); ax.set_ylabel("pérdida (residual² medio)")
ax.legend()
ax.set_title("Entrenamiento de la PINN", loc="left")
guardar(fig, "fig4_perdida_pinn")

# 5. Hiperparametros: residual de validacion vs error real
hp = pd.read_csv("resultados/hiperparametros.csv")
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for ed in EDS:
    s = hp[hp.ED == ed]
    ax.loglog(s.residual_val, s.error_max, marcas[ed], color=tonos[ed], ms=7, label=ed,
              mec="white", mew=1)
ax.set_xlabel("residual de validación (lo que la red puede medir)")
ax.set_ylabel("error máximo real vs. exacta")
ax.legend()
ax.set_title("Menor residual ⇒ menor error: válido para elegir la arquitectura", loc="left",
             fontsize=10)
guardar(fig, "fig5_hiperparametros")

# 6. Extrapolacion (error de la PINN dentro y fuera del dominio)
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for ax, ed in zip(axs, EDS):
    d = pd.read_csv(f"resultados/datos_ext_{ed}.csv")
    T = BANCO[ed]["T"]
    ax.axvspan(T, d.x.max(), color="#f1f0ec", zorder=0)
    ax.text(T + (d.x.max() - T) * 0.05, 0.95, "fuera del\nentrenamiento",
            transform=ax.get_xaxis_transform(), color=TXT2, fontsize=8, va="top")
    ax.semilogy(d.x, np.abs(d.pinn - d.analitico) + 1e-18, color=COL["PINN"], lw=2)
    ax.axvline(T, color=TXT2, lw=1, ls="--")
    ax.set_title(ed); ax.set_xlabel("x")
    ax.set_ylim(1e-9, 1)
axs[0].set_ylabel("|error| de la PINN")
fig.suptitle("Generalización: al salir del dominio entrenado el error de la PINN se dispara",
             x=0.01, y=1.06, ha="left", fontweight="bold", color=TXT)
guardar(fig, "fig6_extrapolacion")

# 7. Costo computacional (escala log)
m = pd.read_csv("resultados/metricas.csv")
fig, ax = plt.subplots(figsize=(8, 4))
conceptos = [
    ("Analítico: resolver simbólicamente (1 corrida)", lambda r: r.t_preparacion_s, "Analítico"),
    ("Analítico: evaluar 501 puntos", lambda r: r.t_evaluacion_s, "Analítico"),
    ("RK4: resolver (500 pasos)", lambda r: r.t_evaluacion_s, "RK4"),
    ("PINN: entrenar (1 corrida)", lambda r: r.t_preparacion_s, "PINN"),
    ("PINN: evaluar 501 puntos", lambda r: r.t_evaluacion_s, "PINN"),
]
etiquetas = [k for k, _, _ in conceptos]
for i, ed in enumerate(EDS):
    for j, (nom, fn, met) in enumerate(conceptos):
        r = m[(m.ED == ed) & (m.metodo == met)].iloc[0]
        ax.semilogx(fn(r), j + (i - 1) * 0.2, marcas[ed], color=tonos[ed], ms=7,
                    label=ed if j == 0 else None, mec="white", mew=1)
ax.set_yticks(range(len(etiquetas)), etiquetas)
ax.invert_yaxis()
ax.set_xlabel("tiempo (segundos, escala log)")
ax.legend(loc="lower right")
ax.set_title("Costo computacional: entrenar la PINN tarda de 800 a 1,500 veces más que RK4 con 500 pasos", loc="left")
guardar(fig, "fig7_costo")

print("Figuras guardadas en figuras/")
