from __future__ import annotations

import os
import re
from urllib import parse
from flask import Flask, render_template, request
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application,
    convert_xor, function_exponentiation
)

app = Flask(__name__)

# Symbols
x, y = sp.symbols("x y", real=True)
C = sp.Symbol("C")  # generic integration constant (display)

TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,                 # allows ^ as power
    function_exponentiation,
)

SAFE_LOCALS = {
    "x": x, "y": y,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "exp": sp.exp, "ln": sp.log, "log": sp.log, "sqrt": sp.sqrt,
    "pi": sp.pi, "E": sp.E, "Abs": sp.Abs}

def parse(user_text: str) -> sp.Expr:
    """
    Parser amigable:
    - Permite multiplicación implícita: 300x^2y, xcos(y), ysen(x)
    - Acepta 'sen' como 'sin'
    - Acepta e^x como exp(x)
    - Potencias: ^ (recomendado) o **
    """
    txt = (user_text or "").strip()
    if not txt:
        raise ValueError("Entrada vacía.")

    # Normalizar espacios
    txt = txt.replace(" ", "")

    # Potencias: permitir ** y ^
    txt = txt.replace("**", "^")

    # Trig en español: sen -> sin (también SEN)
    txt = re.sub(r"(?i)sen(?=\()", "sin", txt)

    # e^algo -> exp(algo)
    # Casos: e^x, e^(x+1), e^2x, e^sin(x)
    txt = re.sub(r"(?i)e\^\(([^)]+)\)", r"exp(\1)", txt)
    txt = re.sub(r"(?i)e\^([A-Za-z0-9_]+)", r"exp(\1)", txt)

    # A veces escriben expx sin paréntesis (opcional); no lo forzamos.

    try:
        return sp.simplify(parse_expr(txt, local_dict=SAFE_LOCALS, transformations=TRANSFORMS, evaluate=True))
    except Exception as e:
        raise ValueError(f"No se pudo interpretar la expresión: {e}")

def normalize_constant(expr: sp.Expr) -> sp.Expr:
    """
    Replace exp(C) or E**C by C (absorbing constants) in common explicit solutions.
    Heuristic: show y = C e^{...} instead of y = e^{C+...}.
    """
    expr = sp.simplify(expr)
    try:
        expr = sp.simplify(sp.factor(expr))
    except Exception:
        pass
    expr = expr.replace(sp.exp(C), C)
    expr = expr.replace(sp.E**C, C)
    return sp.simplify(expr)

def prefer_explicit_y(implicit_eq: sp.Equality) -> sp.Equality:
    """Try to solve implicit equation for y and return y = ... if possible."""
    try:
        sols = sp.solve(implicit_eq, y)
    except Exception:
        return implicit_eq
    if not sols:
        return implicit_eq
    sol = normalize_constant(sols[0])
    return sp.Eq(y, sol)

# ---------------- Routes ----------------
@app.get("/")
def index():
    return render_template("index.html")

@app.route("/separacion", methods=["GET", "POST"])
def separacion():
    resultado_latex = None
    pasos = None
    error = None
    if request.method == "POST":
        try:
            rhs = parse(request.form.get("rhs", ""))
            resultado_latex, pasos = resolver_separacion(rhs)
        except Exception as e:
            error = str(e)
    return render_template("separacion.html", resultado_latex=resultado_latex, pasos=pasos, error=error)

@app.route("/exactas", methods=["GET", "POST"])
def exactas():
    resultado_latex = None
    pasos = None
    error = None
    if request.method == "POST":
        try:
            M = parse(request.form.get("M", ""))
            N = parse(request.form.get("N", ""))
            resultado_latex, pasos = resolver_exacta(M, N)
        except Exception as e:
            error = str(e)
    return render_template("exactas.html", resultado_latex=resultado_latex, pasos=pasos, error=error)


# ---------------- Solvers ----------------
def resolver_separacion(rhs: sp.Expr):
    """
    Solve dy/dx = rhs(x,y) by separation when rhs = f(x)*g(y).
    Prefer a nicer explicit solution y = C e^{...} when possible.
    """
    rhs = sp.simplify(rhs)

    fx, gy = sp.factor_terms(rhs).as_independent(y, as_Add=False)
    fx = sp.simplify(fx)
    gy = sp.simplify(gy)

    if fx.has(y) or gy.has(x):
        fact = sp.factor(rhs)
        fx, gy = sp.factor_terms(fact).as_independent(y, as_Add=False)
        fx = sp.simplify(fx)
        gy = sp.simplify(gy)

    if fx.has(y) or gy.has(x):
        raise ValueError(
            "Esta ecuación no se pudo separar como f(x)·g(y) con el formato actual.\n"
            "Tip: escribe dy/dx = f(x)*g(y) (ej. x*(y+1), (x^2+1)/y, exp(x)*sin(y))."
        )

    if sp.simplify(gy) == 0:
        raise ValueError("La parte g(y) quedó en 0. Revisa la ecuación.")

    lhs_int = sp.integrate(sp.simplify(1/gy), y)
    rhs_int = sp.integrate(fx, x)

    implicit = sp.Eq(lhs_int, rhs_int + C)
    displayed = prefer_explicit_y(implicit)

    steps = [
        {"t": "Entrada", "m": sp.latex(sp.Eq(sp.Derivative(y, x), rhs))},
        {"t": "Identificar", "m": sp.latex(sp.Eq(sp.Symbol("f(x)"), fx)) + r"\\quad " + sp.latex(sp.Eq(sp.Symbol("g(y)"), gy))},
        {"t": "Separar variables", "m": sp.latex(sp.Eq(sp.Mul(1/gy, sp.Symbol("dy")), sp.Mul(fx, sp.Symbol("dx"))))},
        {"t": "Integrar", "m": sp.latex(sp.Eq(sp.Integral(1/gy, y), sp.Integral(fx, x)))},
        {"t": "Resultado (implícito)", "m": sp.latex(implicit)},
    ]
    if displayed != implicit:
        steps.append({"t": "Forma equivalente (más común)", "m": sp.latex(displayed)})

    return sp.latex(displayed), steps

def resolver_exacta(M: sp.Expr, N: sp.Expr):
    M = sp.simplify(M)
    N = sp.simplify(N)

    dM_dy = sp.diff(M, y)
    dN_dx = sp.diff(N, x)

    if sp.simplify(dM_dy - dN_dx) != 0:
        raise ValueError(
            "La ecuación NO es exacta (∂M/∂y ≠ ∂N/∂x).\n"
            "Por ahora este apartado resuelve solo ecuaciones exactas.\n"
            "Si quieres, luego añadimos factor integrante."
        )

    psi_partial = sp.integrate(M, x)
    hprime = sp.simplify(N - sp.diff(psi_partial, y))
    h = sp.integrate(hprime, y)

    psi = sp.simplify(psi_partial + h)
    sol = sp.Eq(psi, C)

    steps = [
        {"t": "Entrada", "m": sp.latex(sp.Eq(M*sp.Symbol("dx") + N*sp.Symbol("dy"), 0))},
        {"t": "Verificar exactitud", "m": sp.latex(sp.Eq(sp.diff(M, y), dM_dy)) + r"\\quad " + sp.latex(sp.Eq(sp.diff(N, x), dN_dx))},
        {"t": "Integrar M en x", "m": r"\Psi(x,y)=\int M\,dx=" + sp.latex(psi_partial) + r"+h(y)"},
        {"t": "Encontrar h'(y)", "m": r"h'(y)=N-\frac{\partial}{\partial y}\left(\int M\,dx\right)=" + sp.latex(hprime)},
        {"t": "Integrar h'(y)", "m": r"h(y)=\int h'(y)\,dy=" + sp.latex(h)},
        {"t": "Solución", "m": sp.latex(sol)},
    ]
    return sp.latex(sol), steps

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)