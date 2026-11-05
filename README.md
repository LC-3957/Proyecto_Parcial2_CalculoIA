# Benchmarking computacional de EDOs: analítico vs. RK4 vs. red neuronal

Proyecto 2do parcial: Cálculo para Inteligencia Artificial, Ingeniería en IA, Ibero León.

## Estructura

| Archivo | Fase | Qué hace |
|---|---|---|
| `edos.py` | - | Banco ED-01, ED-02, ED-03 (ecuación, condiciones, dominio) |
| `analitico.py` | 1 | Coeficientes constantes, variación de parámetros y superposición programados en SymPy, con verificación |
| `rk4.py` | 2 | Runge-Kutta 4 desde cero (sistema de 1er orden) |
| `pinn.py` | 3 | Red neuronal informada por la física (PINN) en PyTorch |
| `benchmark.py` | 4 | Corre todo, calcula MSE / error máximo / costo y guarda `resultados/` y `modelos/` |
| `figuras.py` | 4 | Genera las gráficas del reporte en `figuras/` |
| `app.py` | 0 | Interfaz gráfica (Streamlit) |
| `red_numpy.py` | 0 | Evalúa la red entrenada con NumPy, para que la interfaz no necesite PyTorch |
| `exportar_pesos.py` | 3 | Exporta los pesos de `modelos/*.pt` a `modelos/*.npz` |

## Cómo ejecutarlo

Solo la interfaz (usa los resultados ya incluidos):

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Todo el proyecto (reentrenar la red y regenerar resultados y figuras):

```bash
python -m pip install -r requirements-benchmark.txt
python analitico.py          # solución paso a paso de las 3 EDOs
python benchmark.py          # ~10 min en CPU (python benchmark.py --rapido: ~2 min)
python figuras.py
python exportar_pesos.py     # actualiza los pesos que usa la interfaz
```

## Dominios

- ED-01: [0, 5]
- ED-02: [0, 0.5], porque sec(3x) tiene una asíntota en x = π/6 ≈ 0.5236; la solución no existe más allá.
- ED-03: [0, 1], porque la solución crece como e^{3x}.

Métricas evaluadas en N = 501 puntos equiespaciados de cada dominio.
