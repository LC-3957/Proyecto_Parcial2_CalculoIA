"""Interfaz del proyecto.

Correr con: python -m streamlit run app.py
"""
import json
import re
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sympy as sp

import rk4
from edos import BANCO, EXACTAS, exacta_numpy
from red_numpy import RedNumpy

st.set_page_config(page_title="Benchmark de EDOs", layout="wide", initial_sidebar_state="collapsed")

# paleta
TEXTO = "#22252c"        # tinta
TEXTO_2 = "#5a5f6b"
FONDO = "#f7f4ee"        # papel
LINEA = "#dcd6cb"
COL = {"Analítico": "#2f62b5", "RK4": "#c2562a", "PINN": "#1d7f5f"}       # trazos
PASTEL = {"Analítico": "#e3ebf7", "RK4": "#f8e6dc", "PINN": "#e0f0e8"}    # fondos suaves
OSCURO = "#2b3150"       # azul marino para la barra y títulos
ACENTO = "#6a52c4"       # lila

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&display=swap');
.stApp, .stApp p, .stApp li, .stApp label, .stMarkdown, [data-testid="stSidebar"] p {{
  font-family: 'IBM Plex Sans', system-ui, sans-serif; }}
.stApp {{ background: {FONDO}; color: {TEXTO}; }}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {{ font-family: 'Source Serif 4', Georgia, serif;
  color: {OSCURO}; font-weight: 600; letter-spacing: -0.01em; }}
.stApp h3, .stApp h4 {{ border-left: 4px solid {ACENTO}; padding-left: 10px !important; }}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"] {{ display: none; }}

/* barra superior fija con el selector de ecuación */
div:has(> .st-key-barra) {{ position: sticky; top: 3.75rem; z-index: 999; }}
.st-key-barra {{ position: sticky; top: 3.75rem; z-index: 999; background: {OSCURO};
  border-radius: 6px; padding: 10px 20px; margin-bottom: 22px; box-shadow: 0 4px 14px rgba(43,49,80,.18); }}
.st-key-barra .marca {{ font-family: 'Source Serif 4', Georgia, serif; font-weight: 600; font-size: 1.2rem;
  color: #ffffff; line-height: 1.2; }}
.st-key-barra .marca small {{ display: block; font-family: 'IBM Plex Sans', sans-serif; font-weight: 400;
  font-size: .78rem; color: #c9cde0; margin-top: 2px; }}
.st-key-barra [role="radiogroup"] {{ gap: 10px; flex-wrap: wrap; justify-content: flex-end; }}
/* botones de las ecuaciones */
.st-key-barra [role="radiogroup"] > div:nth-child(1) label,
.st-key-barra label[data-baseweb="radio"]:nth-child(1) {{ background: #dfe8f8; }}
.st-key-barra [role="radiogroup"] > div:nth-child(2) label,
.st-key-barra label[data-baseweb="radio"]:nth-child(2) {{ background: #f9e3d6; }}
.st-key-barra [role="radiogroup"] > div:nth-child(3) label,
.st-key-barra label[data-baseweb="radio"]:nth-child(3) {{ background: #dcefe4; }}
.st-key-barra [data-testid="stRadioOption"], .st-key-barra label[data-baseweb="radio"] {{
  margin: 0; padding: 7px 14px; border-radius: 6px; border: 2px solid transparent; cursor: pointer;
  opacity: .82; transition: opacity .15s, border-color .15s; }}
.st-key-barra [data-testid="stRadioOption"]:hover, .st-key-barra label[data-baseweb="radio"]:hover {{ opacity: 1; }}
.st-key-barra [data-testid="stRadioOption"] > div > div:first-child,
.st-key-barra label[data-baseweb="radio"] > div:first-child {{ display: none; }}
.st-key-barra [data-testid="stRadioOption"] p, .st-key-barra label[data-baseweb="radio"] p {{
  color: {TEXTO} !important; font-size: .92rem; margin: 0; }}
.st-key-barra [data-testid="stRadioOption"][data-selected="true"],
.st-key-barra label[data-baseweb="radio"]:has(input:checked) {{ opacity: 1; border-color: #ffffff; }}
.st-key-barra [data-testid="stRadioOption"][data-selected="true"] p,
.st-key-barra label[data-baseweb="radio"]:has(input:checked) p {{ font-weight: 600; }}
.block-container {{ padding: 3.5rem 2.5rem 2rem; max-width: 100%; }}

.encabezado {{ padding-bottom: 12px; margin-bottom: 14px; border-bottom: 4px solid;
  border-image: linear-gradient(90deg, {COL["Analítico"]} 0 33.3%, {COL["RK4"]} 33.3% 66.6%, {COL["PINN"]} 66.6% 100%) 1; }}
.encabezado .kicker {{ font-size: .78rem; letter-spacing: .12em; text-transform: uppercase; color: {ACENTO}; font-weight: 600; }}
.encabezado .titulo {{ font-family: 'Source Serif 4', Georgia, serif; font-size: 2.3rem; font-weight: 600;
  margin: 4px 0 0; color: {OSCURO}; }}
.ficha {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 0 0 20px; font-size: .95rem; }}
.ficha div {{ padding: 10px 14px; border-radius: 6px; color: {TEXTO}; border-left: 4px solid; }}
.ficha div small {{ display: block; font-size: .72rem; letter-spacing: .1em; text-transform: uppercase;
  color: {TEXTO_2}; margin-bottom: 2px; }}
@media (max-width: 800px) {{ .ficha {{ grid-template-columns: 1fr; }} }}

.tarjeta {{ background: #ffffff; border: 1px solid {LINEA}; border-radius: 6px; padding: 14px 16px 12px; }}
.tarjeta .t {{ font-size: .75rem; letter-spacing: .1em; text-transform: uppercase; font-weight: 600; }}
.tarjeta .v {{ font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem; font-weight: 500; margin: 6px 0 2px; color: {TEXTO}; }}
.tarjeta .s {{ color: {TEXTO_2}; font-size: .86rem; line-height: 1.35; }}

.paso {{ display: flex; align-items: center; gap: 12px; padding: 10px 16px; margin: 8px 0;
  border-radius: 14px; border: 1px solid rgba(43,49,80,.08);
  font-family: 'IBM Plex Mono', monospace; font-size: .9rem; color: {TEXTO}; }}
.paso .n {{ flex: 0 0 26px; height: 26px; border-radius: 50%; background: {OSCURO}; color: #ffffff;
  display: flex; align-items: center; justify-content: center; font-size: .78rem; font-weight: 600; }}
.burbuja {{ border-radius: 14px; padding: 12px 16px; margin: 10px 0; color: {TEXTO};
  border: 1px solid rgba(43,49,80,.08); font-size: .92rem; line-height: 1.5; }}
.burbuja b {{ color: {OSCURO}; }}
.nota {{ font-size: .88rem; color: {TEXTO_2}; border-left: 3px solid {COL["PINN"]}; padding: 4px 0 4px 10px; margin: 10px 0; }}

.stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 2px solid {LINEA}; }}
.stTabs [data-baseweb="tab"] {{ background: transparent; padding: 8px 14px; color: {TEXTO_2}; border-radius: 6px 6px 0 0; }}
.stTabs [data-baseweb="tab"]:hover {{ background: #ece8f5; }}
.stTabs [aria-selected="true"] {{ color: {OSCURO} !important; font-weight: 600; background: #e6e0f5; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {ACENTO}; height: 3px; }}
.stTabs [data-baseweb="tab-border"] {{ display: none; }}
[data-testid="stMetric"] {{ background: #f1edf9; border: 1px solid #ddd5f0; border-left: 4px solid {ACENTO};
  border-radius: 6px; padding: 10px 14px; }}
[data-testid="stMetricLabel"] p {{ color: {TEXTO_2}; }}
[data-testid="stMetricValue"] {{ font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem; }}
[data-testid="stExpander"] details {{ border-radius: 4px; border-color: {LINEA}; background: #ffffff; }}
.stDownloadButton button {{ border-radius: 4px; border: 1px solid {TEXTO}; background: #ffffff; color: {TEXTO}; }}
.stDownloadButton button:hover {{ background: {TEXTO}; color: #ffffff; }}
.pie {{ font-size: .78rem; color: {TEXTO_2}; margin-top: 2rem; }}
</style>
""", unsafe_allow_html=True)


def estilo(fig, titulo="", alto=420, log_y=False):
    fig.update_layout(
        title=dict(text=titulo, x=0, xanchor="left",
                   font=dict(size=15, color=TEXTO, family="Source Serif 4, Georgia, serif")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff",
        font=dict(color=TEXTO, size=12.5, family="IBM Plex Sans, sans-serif"),
        height=alto, hovermode="x unified",
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor=LINEA, borderwidth=1),
        margin=dict(l=60, r=20, t=50, b=45),
    )
    fig.update_xaxes(gridcolor="#efebe4", zerolinecolor=LINEA, linecolor=LINEA, mirror=True, showline=True)
    fig.update_yaxes(gridcolor="#efebe4", zerolinecolor=LINEA, linecolor=LINEA, mirror=True, showline=True)
    if log_y:
        fig.update_yaxes(type="log", exponentformat="power")
    return fig


def tarjeta_neutra(titulo, valor, sub):
    return (f"<div class='tarjeta' style='border-top:3px solid {TEXTO_2}'>"
            f"<div class='t' style='color:{TEXTO_2}'>{titulo}</div>"
            f"<div class='v'>{valor}</div><div class='s'>{sub}</div></div>")


def tarjeta(metodo, valor, sub):
    return (f"<div class='tarjeta' style='border-top:3px solid {COL[metodo]}; background:"
            f"linear-gradient({PASTEL[metodo]}, #ffffff 70%)'>"
            f"<div class='t' style='color:{COL[metodo]}'>{metodo}</div>"
            f"<div class='v'>{valor}</div><div class='s'>{sub}</div></div>")


@st.cache_data
def cargar_csv(nombre):
    return pd.read_csv(f"resultados/{nombre}")


@st.cache_data
def cargar_pasos():
    with open("resultados/analitico.json", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_resource
def cargar_pinn(ed):
    return RedNumpy.cargar(f"modelos/pinn_{ed}.npz", ed)


def fmt(v):
    return f"{v:.2e}" if v != 0 else "0 (referencia)"


def legible(t):
    """Cambia la notación de SymPy por notación normal (r**2 -> r², exp(x) -> e^(x))."""
    t = re.sub(r"exp\(([^()]*)\)", r"e^(\1)", t)
    t = t.replace("**2", "²").replace("**", "^")
    t = re.sub(r"(\d)\*([A-Za-z(])", r"\1\2", t)
    t = re.sub(r"(?<=\d)I\b", "i", t)
    t = t.replace("*", "·").replace("log(", "ln(").replace("->", "→")
    t = t.replace(" (mult. 1)", "")
    t = t.replace(" (mult. 2)", " (raíz doble)")
    t = re.sub(r"(\d+)\.0\b", r"\1", t)
    t = re.sub(r"\)=(\S)", r") = \1", t)
    if t.startswith("Raíces: ") and "i" not in t and "(" not in t:   # raíces reales: de menor a mayor
        try:
            t = "Raíces: " + ", ".join(sorted(t[8:].split(", "), key=float))
        except ValueError:
            pass
    return t


# barra superior fija
with st.container(key="barra"):
    c_marca, c_sel = st.columns([1, 3.2], vertical_alignment="center")
    c_marca.markdown("<div class='marca'>Benchmark de EDOs<small>Analítico, RK4 y red neuronal</small></div>",
                     unsafe_allow_html=True)
    ed = c_sel.radio("Ecuación", list(BANCO), horizontal=True, label_visibility="collapsed",
                     format_func=lambda k: f"{k}:  {BANCO[k]['nombre']}")
p = BANCO[ed]

st.markdown(f"""
<div class='encabezado'>
  <div class='kicker'>Proyecto 2do parcial &nbsp;/&nbsp; {ed}</div>
  <div class='titulo'>{p['nombre']}</div>
</div>
<div class='ficha'>
  <div style='background:#e3ebf7; border-color:{COL["Analítico"]}'><small>Método clásico</small>{p['metodo']}</div>
  <div style='background:#f1edf9; border-color:{ACENTO}'><small>Condiciones iniciales</small>y(0) = {p['y0']:g}, &nbsp;y'(0) = {p['v0']:g}</div>
  <div style='background:#e0f0e8; border-color:{COL["PINN"]}'><small>Dominio</small>[0, {p['T']:g}]</div>
</div>""", unsafe_allow_html=True)
if ed == "ED-02":
    st.markdown("<div class='burbuja' style='background:#f8e6dc'><b>Nota.</b> sec(3x) diverge en "
                "x = π/6 ≈ 0.5236, por eso el dominio se limita a [0, 0.5].</div>", unsafe_allow_html=True)

tabs = st.tabs(["Solución analítica", "Comparación", "Métricas", "RK4 interactivo",
                "Red neuronal", "Conclusiones"])

# Pestaña 1: solución analítica
with tabs[0]:
    pasos = cargar_pasos()[ed]
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader(f"Método: {p['metodo']}")
        lista = pasos["pasos"][1:]
        ver_todo = st.toggle("Mostrar todos los pasos", value=True)
        n = len(lista) if ver_todo else st.slider("Avanza paso a paso", 1, len(lista), 1)
        tonos = ["#e3ebf7", "#f1edf9", "#e0f0e8", "#f8e6dc"]
        for i, s in enumerate(lista[:n], 1):
            st.markdown(f"<div class='paso' style='background:{tonos[(i - 1) % 4]}'>"
                        f"<span class='n'>{i}</span><span>{legible(s)}</span></div>", unsafe_allow_html=True)
    with c2:
        with st.container(border=True):
            st.markdown("**Solución particular del PVI**")
            st.latex("y(x) = " + sp.latex(EXACTAS[ed]))
        st.markdown("<div class='burbuja' style='background:#e0f0e8'><b>Verificada con SymPy.</b> Al "
                    "sustituirla en la ecuación el residual es 0 y cumple las dos condiciones iniciales.</div>",
                    unsafe_allow_html=True)
        xf = np.linspace(0, p["T"], 400)
        f = go.Figure(go.Scatter(x=xf, y=exacta_numpy(ed)(xf), fill="tozeroy",
                                 fillcolor="rgba(47,111,208,0.15)",
                                 line=dict(color=COL["Analítico"], width=3), name="y(x)"))
        f.update_layout(xaxis_title="x", yaxis_title="y(x)")
        st.plotly_chart(estilo(f, "Gráfica de la solución exacta", 320), width="stretch")

# Pestaña 2: comparación
with tabs[1]:
    d = cargar_csv(f"datos_{ed}.csv")
    a, b = st.columns([3, 1])
    with a:
        mostrar = st.multiselect("Métodos a mostrar", ["Analítico", "RK4", "PINN"],
                                 default=["Analítico", "RK4", "PINN"])
    with b:
        log = st.toggle("Error en escala log", value=True)
    xr = st.slider("Acercar a un rango de x", 0.0, float(p["T"]), (0.0, float(p["T"])))
    m = (d.x >= xr[0]) & (d.x <= xr[1])
    dd = d[m]
    c1, c2 = st.columns(2)
    with c1:
        f = go.Figure()
        if "Analítico" in mostrar:
            f.add_scatter(x=dd.x, y=dd.analitico, name="Analítico",
                          line=dict(color=COL["Analítico"], width=3))
        if "RK4" in mostrar:
            f.add_scatter(x=dd.x, y=dd.rk4, name="RK4", line=dict(color=COL["RK4"], width=2, dash="dash"))
        if "PINN" in mostrar:
            paso = max(1, len(dd) // 25)
            f.add_scatter(x=dd.x[::paso], y=dd.pinn[::paso], name="PINN", mode="markers",
                          marker=dict(color=PASTEL["PINN"], size=9, line=dict(color=COL["PINN"], width=2)))
        f.update_layout(xaxis_title="x", yaxis_title="y(x)")
        st.plotly_chart(estilo(f, "Soluciones"), width="stretch")
    with c2:
        f = go.Figure()
        for met, col in [("RK4", "rk4"), ("PINN", "pinn")]:
            if met in mostrar:
                f.add_scatter(x=dd.x, y=np.abs(dd[col] - dd.analitico) + 1e-18, name=met,
                              line=dict(color=COL[met], width=2.5))
        f.update_layout(xaxis_title="x", yaxis_title="|aproximación − exacta|")
        st.plotly_chart(estilo(f, "Error absoluto vs. solución exacta", log_y=log), width="stretch")

# Pestaña 3: métricas
with tabs[2]:
    mt = cargar_csv("metricas.csv")
    s = mt[mt.ED == ed]
    cols = st.columns(3)
    for col, (_, r) in zip(cols, s.iterrows()):
        col.markdown(tarjeta(r.metodo, fmt(r.error_max),
                             f"error máximo · MSE {fmt(r.MSE)}<br>{r.detalle}"), unsafe_allow_html=True)
    st.write("")
    tabla = s[["metodo", "MSE", "error_max", "t_preparacion_s", "t_evaluacion_s", "parametros"]].copy()
    tabla.columns = ["Método", "MSE", "Error máx.", "Preparación (s)", "Evaluación (s)", "Parámetros"]
    tabla[["Preparación (s)", "Parámetros"]] = tabla[["Preparación (s)", "Parámetros"]].astype(float)
    tabla.loc[tabla["Método"] == "RK4", "Preparación (s)"] = np.nan
    tabla.loc[tabla["Parámetros"] == 0, "Parámetros"] = np.nan
    st.dataframe(tabla.style.format({"MSE": "{:.2e}", "Error máx.": "{:.2e}", "Preparación (s)": "{:.3g}",
                                     "Evaluación (s)": "{:.3g}", "Parámetros": "{:,.0f}"}, na_rep="no aplica"),
                 hide_index=True, width="stretch")
    st.markdown("<div class='burbuja' style='background:#f1edf9'><b>Cómo leer la tabla.</b> Preparación es "
                "resolver simbólicamente (analítico) o entrenar (PINN), medido en una corrida. Evaluación es "
                "obtener y(x) en 501 puntos, mediana de varias repeticiones. El error de cada método se mide "
                "contra la solución analítica exacta.</div>", unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    d1.download_button("Descargar métricas de las 3 ecuaciones (CSV)",
                       mt.to_csv(index=False).encode("utf-8"), "metricas.csv", "text/csv")
    d2.download_button(f"Descargar tabla de valores de {ed} (CSV)",
                       cargar_csv(f"datos_{ed}.csv").to_csv(index=False).encode("utf-8"),
                       f"valores_{ed}.csv", "text/csv")

    st.markdown("#### Compara las tres ecuaciones")
    metrica = st.radio("Métrica", ["Error máximo", "MSE", "Tiempo de preparación + evaluación"],
                       horizontal=True)
    campo = {"Error máximo": "error_max", "MSE": "MSE"}.get(metrica)
    f = go.Figure()
    for met in ["RK4", "PINN"]:
        q = mt[mt.metodo == met]
        y = q[campo] if campo else q.t_preparacion_s + q.t_evaluacion_s
        f.add_bar(x=q.ED, y=y, name=met, marker=dict(color=PASTEL[met], line=dict(color=COL[met], width=2)),
                  text=[f"{v:.1e}" for v in y], textposition="outside")
    estilo(f, f"{metrica} por ecuación (escala log)", 400, log_y=True)
    f.update_layout(barmode="group", hovermode="closest", xaxis_title="Ecuación",
                    yaxis_title="segundos" if campo is None else metrica)
    st.plotly_chart(f, width="stretch")

# Pestaña 4: RK4 interactivo
with tabs[3]:
    st.markdown("<div class='burbuja' style='background:#f8e6dc'>Mueve el número de pasos y observa cómo "
                "RK4 se acerca a la solución exacta.</div>", unsafe_allow_html=True)
    n = st.slider("Número de pasos de RK4", 2, 400, 20)
    t0 = time.perf_counter()
    xr_, yr = rk4.resolver(ed, n)
    t_rk = time.perf_counter() - t0
    xf = np.linspace(0, p["T"], 600)
    fe = exacta_numpy(ed)
    err = np.max(np.abs(yr - fe(xr_)))
    e_pinn = cargar_csv("metricas.csv").query("ED == @ed and metodo == 'PINN'").error_max.iloc[0]
    a, b, c = st.columns(3)
    a.metric("Tamaño de paso h", f"{p['T'] / n:.4g}")
    b.metric("Error máximo", f"{err:.2e}",
             "mejor que la PINN" if err <= e_pinn else "peor que la PINN",
             delta_color="normal" if err <= e_pinn else "inverse")
    c.metric("Tiempo", f"{t_rk * 1e3:.2f} ms")
    orden = int(np.floor(np.log10(max(err, 1e-16))))
    st.progress(float(np.clip(-orden / 12, 0, 1)),
                text=f"El error máximo es del orden de 10^{orden} (la barra se llena al llegar a 10^-12)")
    f = go.Figure()
    f.add_scatter(x=xf, y=fe(xf), name="Analítico", line=dict(color=COL["Analítico"], width=3))
    f.add_scatter(x=xr_, y=yr, name=f"RK4 (n={n})", mode="lines+markers",
                  line=dict(color=COL["RK4"], width=2), marker=dict(size=7, color=PASTEL["RK4"],
                                                                     line=dict(color=COL["RK4"], width=2)))
    f.update_layout(xaxis_title="x", yaxis_title="y(x)")
    st.plotly_chart(estilo(f, "RK4 contra la solución exacta"), width="stretch")
    tabla_rk = pd.DataFrame({"x": xr_, "y_RK4": yr, "y_exacta": fe(xr_), "error_abs": np.abs(yr - fe(xr_))})
    with st.expander(f"Ver tabla de RK4 con n = {n}"):
        st.dataframe(tabla_rk.style.format({"x": "{:.4f}", "y_RK4": "{:.8f}", "y_exacta": "{:.8f}",
                                            "error_abs": "{:.2e}"}), hide_index=True, width="stretch")
        st.download_button("Descargar esta tabla (CSV)", tabla_rk.to_csv(index=False).encode("utf-8"),
                           f"rk4_{ed}_n{n}.csv", "text/csv")
    conv = cargar_csv("convergencia_rk4.csv").query("ED == @ed")
    f = go.Figure()
    f.add_scatter(x=conv.pasos, y=conv.error_max, mode="lines+markers", name="RK4",
                  line=dict(color=COL["RK4"], width=2.5), marker=dict(size=8))
    ref = conv.error_max.iloc[2] * (conv.pasos / conv.pasos.iloc[2]) ** -4
    f.add_scatter(x=conv.pasos, y=ref, name="referencia O(h⁴)", line=dict(color="#8a8f9c", dash="dot"))
    f.add_hline(y=e_pinn, line=dict(color=COL["PINN"], dash="dash"),
                annotation_text="error de la PINN", annotation_font_color=COL["PINN"])
    f.add_vline(x=n, line=dict(color="#b9a8e6", width=2), annotation_text=f"n elegido = {n}")
    estilo(f, "Convergencia: error máximo vs. pasos", 400, log_y=True)
    f.update_xaxes(type="log", title="número de pasos n")
    f.update_yaxes(title="error máximo")
    st.plotly_chart(f, width="stretch")

# Pestaña 5: red neuronal
with tabs[4]:
    modelo = cargar_pinn(ed)
    a, b, c = st.columns(3)
    a.metric("Arquitectura", f"{modelo.capas} capas × {modelo.ancho}")
    b.metric("Parámetros", f"{modelo.n_parametros():,}")
    c.metric("Entrenamiento", f"{modelo.t_entrenamiento:.1f} s")
    st.latex(r"\hat y(x) = y_0 + y_0'\,x + x^2\,N_\theta(x/T),\qquad "
             r"\mathcal{L}(\theta)=\frac{1}{N_c}\sum_i\big(a\hat y''+b\hat y'+c\hat y-g(x_i)\big)^2")

    st.markdown("#### Evalúa la red en cualquier punto")
    x_usr = st.slider("x", 0.0, float(p["T"]), float(p["T"]) / 3, step=float(p["T"]) / 200)
    yv, ye = modelo.predecir([x_usr])[0], float(exacta_numpy(ed)(x_usr))
    a, b, c = st.columns(3)
    a.markdown(tarjeta("PINN", f"{yv:.6f}", "predicción de la red"), unsafe_allow_html=True)
    b.markdown(tarjeta("Analítico", f"{ye:.6f}", "solución exacta"), unsafe_allow_html=True)
    c.markdown(tarjeta_neutra("Error en ese punto", f"{abs(yv - ye):.1e}", "|PINN − exacta|"),
               unsafe_allow_html=True)
    st.write("")

    c1, c2 = st.columns(2)
    with c1:
        h = cargar_csv(f"perdida_{ed}.csv")
        f = go.Figure()
        for fase, color, nom in [("adam", COL["PINN"], "Adam"), ("lbfgs", COL["Analítico"], "L-BFGS")]:
            q = h[h.fase == fase]
            f.add_scatter(x=q.iteracion, y=q.perdida, name=nom, line=dict(color=color, width=2.5))
        f.update_layout(xaxis_title="iteración", yaxis_title="pérdida (residual² medio)")
        st.plotly_chart(estilo(f, "Curva de pérdida", 380, log_y=True), width="stretch")
    with c2:
        de = cargar_csv(f"datos_ext_{ed}.csv")
        f = go.Figure()
        f.add_vrect(x0=p["T"], x1=de.x.max(), fillcolor="#fde3d4", opacity=0.6, line_width=0,
                    annotation_text="fuera del entrenamiento", annotation_position="top left")
        f.add_scatter(x=de.x, y=np.abs(de.pinn - de.analitico) + 1e-18, name="|error| PINN",
                      line=dict(color=COL["PINN"], width=2.5))
        f.update_layout(xaxis_title="x", yaxis_title="|PINN − exacta|")
        st.plotly_chart(estilo(f, "Generalización: error fuera del dominio", 380, log_y=True),
                        width="stretch")

    with st.expander("Barrido de hiperparámetros y robustez"):
        st.markdown("<div class='burbuja' style='background:#e3ebf7'>La arquitectura se elige por residual de "
                    "validación, sin usar la solución exacta.</div>", unsafe_allow_html=True)
        hp = cargar_csv("hiperparametros.csv").query("ED == @ed").drop(columns="ED")
        hp.columns = ["Capas", "Neuronas por capa", "Parámetros", "Residual de validación", "MSE",
                      "Error máx.", "Entrenamiento (s)"]
        st.dataframe(hp.style.format({"Residual de validación": "{:.2e}", "MSE": "{:.2e}", "Error máx.": "{:.2e}",
                                      "Entrenamiento (s)": "{:.1f}", "Parámetros": "{:,}"}),
                     hide_index=True, width="stretch")
        sem = cargar_csv("semillas.csv").query("ED == @ed")
        st.markdown(f"**Robustez con {len(sem)} semillas aleatorias:** el error máximo fue de "
                    f"{sem.error_max.min():.1e} a {sem.error_max.max():.1e} (mediana {sem.error_max.median():.1e}).")

# Pestaña 6: conclusiones
with tabs[5]:
    eq = cargar_csv("equivalencia.csv")
    ext = cargar_csv("extrapolacion.csv")
    r = eq[eq.ED == ed].iloc[0]
    a, b, c = st.columns(3)
    a.markdown(tarjeta("PINN", f"{r.error_max_PINN:.1e}", "error máximo de la red"), unsafe_allow_html=True)
    b.markdown(tarjeta("RK4", f"{int(r.pasos_RK4_equivalentes)} pasos",
                       "bastan para igualar a la red"), unsafe_allow_html=True)
    c.markdown(tarjeta_neutra("Costo relativo", f"{r.factor_tiempo:,.0f} veces",
                              "más tiempo para entrenar la red que para correr ese RK4"),
               unsafe_allow_html=True)
    st.write("")
    with st.expander("Tablas completas"):
        st.markdown("**Pasos de RK4 necesarios para igualar el error de la PINN**")
        eq2 = eq.copy()
        eq2.columns = ["Ecuación", "Error máx. PINN", "Pasos de RK4", "Error máx. RK4", "Tiempo RK4 (s)",
                       "Entrenamiento PINN (s)", "Veces más lento entrenar"]
        st.dataframe(eq2.style.format({"Error máx. PINN": "{:.2e}", "Error máx. RK4": "{:.2e}",
                                       "Tiempo RK4 (s)": "{:.2e}", "Entrenamiento PINN (s)": "{:.1f}",
                                       "Veces más lento entrenar": "{:,.0f}"}), hide_index=True, width="stretch")
        st.markdown("**Error fuera del dominio de entrenamiento**")
        ext2 = ext[["ED", "intervalo", "error_max_PINN_en_dominio", "error_max_PINN", "error_max_RK4"]].copy()
        ext2.columns = ["Ecuación", "Intervalo extrapolado", "PINN dentro del dominio", "PINN fuera",
                        "RK4 fuera"]
        st.dataframe(ext2.style.format({c: "{:.2e}" for c in ext2.columns[2:]}), hide_index=True, width="stretch")
    st.markdown("#### Respuesta a la pregunta del proyecto")
    st.markdown(
        "<div class='burbuja' style='background:#e3ebf7'>La red neuronal sí puede aproximar la solución con "
        "buena precisión: en las tres ecuaciones alcanzó errores máximos entre 1.4×10⁻⁶ y 9.0×10⁻⁵ usando solo "
        "la ecuación, sin ningún dato de la solución. Sin embargo, RK4 consigue ese mismo error con apenas 26 a "
        "42 pasos y en menos de 2 ms, mientras que entrenar la red tomó entre 9.1 y 10.4 segundos.</div>"
        "<div class='burbuja' style='background:#e0f0e8'>Por eso, para ecuaciones ordinarias pequeñas con "
        "condiciones iniciales conocidas, RK4 es la mejor opción. La red conviene cuando se necesita una solución "
        "continua que se evaluará muchas veces, cuando hay datos experimentales o parámetros desconocidos, o en "
        "problemas de muchas variables, donde una malla numérica se vuelve demasiado costosa.</div>"
        "<div class='burbuja' style='background:#f8e6dc'>Fuera del intervalo donde se entrenó, su error crece "
        "entre 106 y 1,177 veces, así que no sirve para extrapolar.</div>", unsafe_allow_html=True)

st.markdown("<div class='pie'>Cálculo para Inteligencia Artificial · Ingeniería en Inteligencia Artificial, "
            "Universidad Iberoamericana León</div>", unsafe_allow_html=True)
