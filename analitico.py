"""Solución analítica de las tres EDOs con SymPy.

Coeficientes constantes, variación de parámetros y superposición.
Cada función regresa y(x) y la lista de pasos que muestra la interfaz.
"""
import sympy as sp

from edos import BANCO, EXACTAS, x

C1, C2 = sp.symbols("C1 C2")


# Solución homogénea (la usan los tres métodos)
def base_homogenea(a, b, c, pasos):
    r = sp.symbols("r")
    car = a * r**2 + b * r + c
    raices = sp.roots(sp.Poly(car, r))
    pasos.append(f"Ecuación característica: {sp.sstr(car)} = 0")
    orden = sorted(raices.items(), key=lambda km: (float(sp.re(km[0])), float(sp.im(km[0]))))
    pasos.append(f"Raíces: {', '.join(f'{sp.sstr(k)} (mult. {m})' for k, m in orden)}")

    lista = list(raices.items())
    if len(lista) == 1:  # raíz real repetida
        r1, _ = lista[0]
        y1, y2 = sp.exp(r1 * x), x * sp.exp(r1 * x)
        pasos.append("Caso: raíz real repetida -> y1 = e^(rx), y2 = x e^(rx)")
    else:
        r1, r2 = lista[0][0], lista[1][0]
        if sp.im(r1) != 0:  # complejas conjugadas
            al, be = sp.re(r1), abs(sp.im(r1))
            y1, y2 = sp.exp(al * x) * sp.cos(be * x), sp.exp(al * x) * sp.sin(be * x)
            pasos.append(f"Caso: raíces complejas α ± βi con α={al}, β={be}")
        else:
            r1, r2 = sorted([r1, r2], key=lambda v: float(v))
            y1, y2 = sp.exp(r1 * x), sp.exp(r2 * x)
            pasos.append("Caso: raíces reales distintas -> y1 = e^(r1 x), y2 = e^(r2 x)")
    pasos.append(f"y_h = C1·({sp.sstr(y1)}) + C2·({sp.sstr(y2)})")
    return sp.simplify(y1), sp.simplify(y2)


def aplicar_condiciones(y_gen, y0, v0, pasos):
    ecs = [
        sp.Eq(y_gen.subs(x, 0), sp.nsimplify(y0)),
        sp.Eq(sp.diff(y_gen, x).subs(x, 0), sp.nsimplify(v0)),
    ]
    sol = sp.solve(ecs, [C1, C2], dict=True)[0]
    pasos.append(f"Condiciones iniciales: y(0)={y0}, y'(0)={v0} -> C1 = {sol[C1]}, C2 = {sol[C2]}")
    return sp.expand(y_gen.subs(sol))


# Coeficientes constantes
def coeficientes_constantes(p):
    pasos = ["MÉTODO: coeficientes constantes (ED homogénea)"]
    y1, y2 = base_homogenea(p["a"], p["b"], p["c"], pasos)
    y = aplicar_condiciones(C1 * y1 + C2 * y2, p["y0"], p["v0"], pasos)
    return y, pasos


# Variación de parámetros
def variacion_parametros(p):
    pasos = ["MÉTODO: variación de parámetros"]
    g = p["g"] / p["a"]  # forma estándar: se divide entre a
    y1, y2 = base_homogenea(p["a"], p["b"], p["c"], pasos)
    W = sp.simplify(sp.Matrix([[y1, y2], [sp.diff(y1, x), sp.diff(y2, x)]]).det())
    pasos.append(f"Wronskiano W(y1, y2) = {sp.sstr(W)}")
    du1 = sp.simplify(-y2 * g / W)
    du2 = sp.simplify(y1 * g / W)
    pasos.append(f"u1' = -y2·g/W = {sp.sstr(du1)}")
    pasos.append(f"u2' =  y1·g/W = {sp.sstr(du2)}")
    u1 = sp.integrate(du1, x)
    u2 = sp.integrate(du2, x)
    # en el dominio cos(3x) > 0, se quita el valor absoluto
    u1 = u1.replace(sp.Abs, lambda arg: arg)
    pasos.append(f"u1 = ∫u1' dx = {sp.sstr(u1)}")
    pasos.append(f"u2 = ∫u2' dx = {sp.sstr(u2)}")
    yp = sp.simplify(u1 * y1 + u2 * y2)
    pasos.append(f"y_p = u1·y1 + u2·y2 = {sp.sstr(yp)}")
    y = aplicar_condiciones(C1 * y1 + C2 * y2 + yp, p["y0"], p["v0"], pasos)
    return y, pasos


# Superposición
def _particular_un_termino(a, b, c, gi, y1, y2):
    """Coeficientes indeterminados para un término k*e^(mx) o k*sin(wx), k*cos(wx)."""
    A, B = sp.symbols("A B")
    gi_e = sp.expand(gi)
    if gi_e.has(sp.exp):
        m = sp.Wild("m")
        mm = [t for t in gi_e.atoms(sp.exp)][0].args[0].match(m * x)[m]
        trial = A * sp.exp(mm * x)
        incog = [A]
    else:
        w = sp.Wild("w")
        f = [t for t in gi_e.atoms(sp.sin, sp.cos)][0]
        ww = f.args[0].match(w * x)[w]
        trial = A * sp.cos(ww * x) + B * sp.sin(ww * x)
        incog = [A, B]
    # si el ensayo ya es solución de la homogénea se multiplica por x
    s = 0
    while sp.simplify(a * sp.diff(trial, x, 2) + b * sp.diff(trial, x) + c * trial).subs(
        {k: 1 for k in incog}
    ) == 0:
        trial, s = x * trial, s + 1
    L = a * sp.diff(trial, x, 2) + b * sp.diff(trial, x) + c * trial - gi
    # se evalúa en varios puntos para sacar los coeficientes
    ecs = [sp.Eq(L.subs(x, xv), 0) for xv in [0, 1, 2][: len(incog)]]
    sol = sp.solve(ecs, incog, dict=True)[0]
    return sp.expand(trial.subs(sol)), trial


def superposicion(p):
    pasos = ["MÉTODO: superposición (cada término por coeficientes indeterminados)"]
    a, b, c = p["a"], p["b"], p["c"]
    y1, y2 = base_homogenea(a, b, c, pasos)
    terminos = sp.Add.make_args(sp.expand(p["g"]))
    pasos.append(f"g(x) se separa en {len(terminos)} términos: " + " | ".join(sp.sstr(t) for t in terminos))
    yp = 0
    for i, gi in enumerate(terminos, 1):
        ypi, ensayo = _particular_un_termino(a, b, c, gi, y1, y2)
        pasos.append(f"  L[y_p{i}] = {sp.sstr(gi)}: ensayo {sp.sstr(ensayo)} -> y_p{i} = {sp.sstr(ypi)}")
        yp += ypi
    pasos.append(f"Principio de superposición: y_p = Σ y_pi = {sp.sstr(yp)}")
    y = aplicar_condiciones(C1 * y1 + C2 * y2 + yp, p["y0"], p["v0"], pasos)
    return y, pasos


METODOS = {
    "Coeficientes constantes": coeficientes_constantes,
    "Variación de parámetros": variacion_parametros,
    "Superposición": superposicion,
}


def verificar(ed_id, y):
    """Revisa que el residual sea 0, las condiciones iniciales y que coincida con la solución a mano."""
    p = BANCO[ed_id]
    res = sp.simplify(p["a"] * sp.diff(y, x, 2) + p["b"] * sp.diff(y, x) + p["c"] * y - p["g"])
    ci0 = sp.simplify(y.subs(x, 0) - sp.nsimplify(p["y0"]))
    ci1 = sp.simplify(sp.diff(y, x).subs(x, 0) - sp.nsimplify(p["v0"]))
    igual_exacta = sp.simplify(y - EXACTAS[ed_id]) == 0
    return dict(residual=res, ci_y=ci0, ci_dy=ci1, coincide_con_mano=igual_exacta)


def resolver(ed_id):
    p = BANCO[ed_id]
    y, pasos = METODOS[p["metodo"]](p)
    return y, pasos


if __name__ == "__main__":
    for ed in BANCO:
        y, pasos = resolver(ed)
        print(f"\n===== {ed}: {BANCO[ed]['nombre']} =====")
        for s in pasos:
            print("  " + s)
        print(f"  y(x) = {y}")
        print("  Verificación:", verificar(ed, y))
