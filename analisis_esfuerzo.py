# =============================================================================
# PROYECTO FINAL: ESTIMACIÓN DE ESFUERZO DE SOFTWARE
# Técnicas de Modelado Predictivo y Evaluación Estadística
#
# Alumno : Guillermo Velázquez Rosiles
# Matrícula: zS23014116
# Asignatura: Técnicas de Modelado Predictivo y Evaluación Estadística
# Fecha   : Junio 2026
# =============================================================================
# DEPENDENCIAS:
#   pip install pandas numpy matplotlib seaborn scipy scikit-learn
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ── Carpeta de salida para imágenes ──────────────────────────────────────────
OUTPUT_DIR = "figuras"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Estilo global de matplotlib (fondo blanco, legible)
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.4,
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
})


# =============================================================================
# CARGA DE DATOS
# =============================================================================
df_A = pd.read_csv("429.csv")   # Dataset A – 429 registros
df_B = pd.read_csv("45.csv")    # Dataset B –  45 registros

print("=" * 60)
print("DATOS CARGADOS")
print(f"  Dataset A (429.csv): {df_A.shape[0]} filas × {df_A.shape[1]} columnas")
print(f"  Dataset B ( 45.csv): {df_B.shape[0]} filas × {df_B.shape[1]} columnas")
print("=" * 60)


# =============================================================================
# FASE I — ANÁLISIS EXPLORATORIO DE DATOS (AED)
# =============================================================================
print("\n" + "=" * 60)
print("FASE I: ANÁLISIS EXPLORATORIO DE DATOS")
print("=" * 60)

# ── 1.1 Tabla de estadísticos descriptivos ───────────────────────────────────
def tabla_descriptivos(df, nombre):
    """Calcula estadísticos descriptivos para FSM y Effort."""
    filas = []
    for col in ["FSM", "Effort"]:
        s = df[col]
        filas.append({
            "Dataset":  nombre,
            "Variable": col,
            "Media":    round(s.mean(), 2),
            "Mediana":  round(s.median(), 2),
            "Desv. Std": round(s.std(), 2),
            "Mínimo":   round(s.min(), 2),
            "Q1":       round(s.quantile(0.25), 2),
            "Q3":       round(s.quantile(0.75), 2),
            "Máximo":   round(s.max(), 2),
        })
    return pd.DataFrame(filas)

desc_A = tabla_descriptivos(df_A, "A (n=429)")
desc_B = tabla_descriptivos(df_B, "B (n=45)")
desc_total = pd.concat([desc_A, desc_B], ignore_index=True)

print("\nTabla 1: Estadísticos descriptivos")
print(desc_total.to_string(index=False))


# ── 1.2 Histogramas ──────────────────────────────────────────────────────────
def plot_histogramas(df, nombre_ds, tag):
    """Genera histogramas para FSM y Effort de un dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Histogramas – Dataset {nombre_ds}", fontweight="bold")

    for ax, col, color in zip(axes, ["FSM", "Effort"], ["steelblue", "darkorange"]):
        ax.hist(df[col], bins=20, color=color, edgecolor="white", alpha=0.85)
        ax.set_xlabel(col)
        ax.set_ylabel("Frecuencia")
        ax.set_title(f"Distribución de {col}")

    plt.tight_layout()
    ruta = os.path.join(OUTPUT_DIR, f"hist_{tag}.png")
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {ruta}")

plot_histogramas(df_A, "A (n=429)", "A")
plot_histogramas(df_B, "B (n=45)",  "B")


# ── 1.3 Diagramas de caja (boxplots) ────────────────────────────────────────
def plot_boxplots(df, nombre_ds, tag):
    """Genera boxplots para FSM y Effort de un dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"Boxplots – Dataset {nombre_ds}", fontweight="bold")

    for ax, col, color in zip(axes, ["FSM", "Effort"], ["steelblue", "darkorange"]):
        ax.boxplot(df[col], patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.6),
                   medianprops=dict(color="black", linewidth=2))
        ax.set_xlabel(col)
        ax.set_ylabel("Valor")
        ax.set_title(f"Boxplot de {col}")

    plt.tight_layout()
    ruta = os.path.join(OUTPUT_DIR, f"box_{tag}.png")
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {ruta}")

plot_boxplots(df_A, "A (n=429)", "A")
plot_boxplots(df_B, "B (n=45)",  "B")


# ── 1.4 Gráficos de dispersión con línea de tendencia ───────────────────────
def plot_scatter(df, nombre_ds, tag):
    """Genera scatter plot NESMA vs Esfuerzo con línea de tendencia."""
    x = df["FSM"].values
    y = df["Effort"].values

    # Línea de tendencia (mínimos cuadrados)
    m, b, r, p, se = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 300)
    y_line = m * x_line + b

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, alpha=0.5, color="steelblue", edgecolors="none", s=18,
               label="Observaciones")
    ax.plot(x_line, y_line, color="crimson", linewidth=1.8,
            label=f"Tendencia: y = {m:.2f}x + {b:.2f}")
    ax.set_xlabel("FSM – Tamaño Funcional (puntos de función NESMA)")
    ax.set_ylabel("Effort – Esfuerzo (horas-hombre)")
    ax.set_title(f"Dispersión FSM vs. Esfuerzo – Dataset {nombre_ds}")
    ax.legend()
    plt.tight_layout()

    ruta = os.path.join(OUTPUT_DIR, f"scatter_{tag}.png")
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {ruta}")

    return r, r**2

r_A, r2_A_simple = plot_scatter(df_A, "A (n=429)", "A")
r_B, r2_B_simple = plot_scatter(df_B, "B (n=45)",  "B")

print(f"\n  Correlación de Pearson – Dataset A: r = {r_A:.4f}")
print(f"  Correlación de Pearson – Dataset B: r = {r_B:.4f}")


# =============================================================================
# FASE II — REGRESIÓN LINEAL SIMPLE
# =============================================================================
print("\n" + "=" * 60)
print("FASE II: REGRESIÓN LINEAL SIMPLE")
print("=" * 60)

def regresion_lineal(df, nombre_ds, tag):
    """
    Ajusta un modelo de regresión lineal simple y genera gráficos de residuos.
    Devuelve un dict con métricas y los residuos.
    """
    X = df[["FSM"]].values
    y = df["Effort"].values

    modelo = LinearRegression()
    modelo.fit(X, y)
    y_pred = modelo.predict(X)
    residuos = y - y_pred

    r2    = r2_score(y, y_pred)
    rmse  = np.sqrt(mean_squared_error(y, y_pred))
    mae   = mean_absolute_error(y, y_pred)
    b0    = modelo.intercept_
    b1    = modelo.coef_[0]

    print(f"\n  Dataset {nombre_ds}")
    print(f"    Ecuación: Effort = {b1:.4f} × FSM + ({b0:.4f})")
    print(f"    R²   = {r2:.4f}")
    print(f"    RMSE = {rmse:.4f}")
    print(f"    MAE  = {mae:.4f}")

    # ── Gráfico 1: Residuos vs valores ajustados ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Análisis de Residuos – Dataset {nombre_ds}", fontweight="bold")

    axes[0].scatter(y_pred, residuos, alpha=0.5, color="steelblue",
                    edgecolors="none", s=18)
    axes[0].axhline(0, color="crimson", linewidth=1.5, linestyle="--")
    axes[0].set_xlabel("Valores Ajustados (ŷ)")
    axes[0].set_ylabel("Residuos (y − ŷ)")
    axes[0].set_title("Residuos vs. Valores Ajustados")

    # ── Gráfico 2: Q-Q plot de residuos ───────────────────────────────────
    stats.probplot(residuos, dist="norm", plot=axes[1])
    axes[1].set_title("Gráfico Q-Q de Residuos")
    axes[1].get_lines()[0].set(color="steelblue", markersize=3, alpha=0.6)
    axes[1].get_lines()[1].set(color="crimson", linewidth=1.5)

    plt.tight_layout()
    ruta = os.path.join(OUTPUT_DIR, f"residuos_{tag}.png")
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Gráfico de residuos guardado: {ruta}")

    return {"b0": b0, "b1": b1, "r2": r2, "rmse": rmse, "mae": mae,
            "residuos": residuos, "y_pred": y_pred, "y_real": y}

metricas_A = regresion_lineal(df_A, "A (n=429)", "A")
metricas_B = regresion_lineal(df_B, "B (n=45)",  "B")


# =============================================================================
# FASE III — EVALUACIÓN ESTADÍSTICA DE MODELOS
# =============================================================================
print("\n" + "=" * 60)
print("FASE III: EVALUACIÓN ESTADÍSTICA DE MODELOS")
print("=" * 60)

# ── 3.1 Pruebas de normalidad sobre los residuales ───────────────────────────
def pruebas_normalidad(residuos, nombre_ds):
    """
    Aplica pruebas de Shapiro-Wilk y Kolmogorov-Smirnov sobre los residuos.
    Documenta H0, H1, estadístico, p-valor e interpretación.
    """
    print(f"\n  Dataset {nombre_ds}")

    # Prueba 1: Shapiro-Wilk
    stat_sw, p_sw = stats.shapiro(residuos)
    print(f"\n    [Prueba 1] Shapiro-Wilk")
    print(f"      H0: Los residuos siguen una distribución normal.")
    print(f"      H1: Los residuos NO siguen una distribución normal.")
    print(f"      Estadístico W = {stat_sw:.6f}")
    print(f"      p-valor       = {p_sw:.6f}")
    interp_sw = "No se rechaza H0 (normalidad)." if p_sw > 0.05 else "Se RECHAZA H0 (no hay normalidad)."
    print(f"      Interpretación: {interp_sw}")

    # Prueba 2: Kolmogorov-Smirnov (vs normal estándar normalizada)
    res_norm = (residuos - residuos.mean()) / residuos.std()
    stat_ks, p_ks = stats.kstest(res_norm, "norm")
    print(f"\n    [Prueba 2] Kolmogorov-Smirnov (KS)")
    print(f"      H0: Los residuos siguen una distribución normal.")
    print(f"      H1: Los residuos NO siguen una distribución normal.")
    print(f"      Estadístico D = {stat_ks:.6f}")
    print(f"      p-valor       = {p_ks:.6f}")
    interp_ks = "No se rechaza H0 (normalidad)." if p_ks > 0.05 else "Se RECHAZA H0 (no hay normalidad)."
    print(f"      Interpretación: {interp_ks}")

    normalidad = (p_sw > 0.05) and (p_ks > 0.05)
    return normalidad, stat_sw, p_sw, stat_ks, p_ks

norm_A, sw_stat_A, sw_p_A, ks_stat_A, ks_p_A = pruebas_normalidad(
    metricas_A["residuos"], "A (n=429)")
norm_B, sw_stat_B, sw_p_B, ks_stat_B, ks_p_B = pruebas_normalidad(
    metricas_B["residuos"], "B (n=45)")


# ── 3.2 Histogramas de residuos ───────────────────────────────────────────────
def plot_hist_residuos(residuos, nombre_ds, tag):
    """Histograma de residuos con curva normal superpuesta."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuos, bins=25, density=True, color="steelblue",
            edgecolor="white", alpha=0.8, label="Residuos")
    # Curva normal teórica
    mu, sigma = residuos.mean(), residuos.std()
    x_n = np.linspace(residuos.min(), residuos.max(), 300)
    ax.plot(x_n, stats.norm.pdf(x_n, mu, sigma), color="crimson",
            linewidth=2, label="Normal teórica")
    ax.set_xlabel("Residuo (y − ŷ)")
    ax.set_ylabel("Densidad")
    ax.set_title(f"Distribución de Residuos – Dataset {nombre_ds}")
    ax.legend()
    plt.tight_layout()
    ruta = os.path.join(OUTPUT_DIR, f"hist_residuos_{tag}.png")
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {ruta}")

plot_hist_residuos(metricas_A["residuos"], "A (n=429)", "A")
plot_hist_residuos(metricas_B["residuos"], "B (n=45)",  "B")


# ── 3.3 Comparación de rendimiento entre datasets ────────────────────────────
print("\n  ── Comparación de rendimiento entre datasets ──")
ambos_normales = norm_A and norm_B

if ambos_normales:
    # ANOVA de un factor sobre los residuales absolutos (como proxy de error)
    stat_cmp, p_cmp = stats.f_oneway(
        np.abs(metricas_A["residuos"]),
        np.abs(metricas_B["residuos"])
    )
    metodo_cmp = "ANOVA (F)"
    print(f"\n  Ambos datasets cumplen normalidad → se aplica ANOVA.")
else:
    # Prueba de Mann-Whitney U (equivalente no paramétrico para 2 grupos)
    stat_cmp, p_cmp = stats.mannwhitneyu(
        np.abs(metricas_A["residuos"]),
        np.abs(metricas_B["residuos"]),
        alternative="two-sided"
    )
    metodo_cmp = "Mann-Whitney U"
    print(f"\n  Al menos un dataset NO cumple normalidad → se aplica Mann-Whitney U.")

print(f"    Método: {metodo_cmp}")
print(f"    H0: No existen diferencias significativas en el error entre datasets.")
print(f"    H1: Existen diferencias significativas en el error entre datasets.")
print(f"    Estadístico = {stat_cmp:.4f}")
print(f"    p-valor     = {p_cmp:.6f}")

if p_cmp < 0.05:
    ganador = "A" if np.abs(metricas_A["residuos"]).mean() < np.abs(metricas_B["residuos"]).mean() else "B"
    print(f"    → Se RECHAZA H0: hay diferencias significativas. Mejor desempeño: Dataset {ganador}.")
else:
    print(f"    → No se rechaza H0: no hay diferencias estadísticamente significativas.")


# ── 3.4 Tabla comparativa de métricas ────────────────────────────────────────
tabla_metricas = pd.DataFrame([
    {"Dataset": "A (n=429)",
     "b0 (intercepto)": round(metricas_A["b0"], 4),
     "b1 (pendiente)":  round(metricas_A["b1"], 4),
     "R²":   round(metricas_A["r2"],   4),
     "RMSE": round(metricas_A["rmse"], 4),
     "MAE":  round(metricas_A["mae"],  4)},
    {"Dataset": "B (n=45)",
     "b0 (intercepto)": round(metricas_B["b0"], 4),
     "b1 (pendiente)":  round(metricas_B["b1"], 4),
     "R²":   round(metricas_B["r2"],   4),
     "RMSE": round(metricas_B["rmse"], 4),
     "MAE":  round(metricas_B["mae"],  4)},
])
print("\n\nTabla 2: Métricas de rendimiento de los modelos de regresión")
print(tabla_metricas.to_string(index=False))


# ── 3.5 Tabla de pruebas de normalidad ───────────────────────────────────────
tabla_norm = pd.DataFrame([
    {"Dataset": "A (n=429)", "Prueba": "Shapiro-Wilk",
     "Estadístico": round(sw_stat_A, 6), "p-valor": round(sw_p_A, 6),
     "¿Normal? (α=0.05)": "Sí" if sw_p_A > 0.05 else "No"},
    {"Dataset": "A (n=429)", "Prueba": "Kolmogorov-Smirnov",
     "Estadístico": round(ks_stat_A, 6), "p-valor": round(ks_p_A, 6),
     "¿Normal? (α=0.05)": "Sí" if ks_p_A > 0.05 else "No"},
    {"Dataset": "B (n=45)",  "Prueba": "Shapiro-Wilk",
     "Estadístico": round(sw_stat_B, 6), "p-valor": round(sw_p_B, 6),
     "¿Normal? (α=0.05)": "Sí" if sw_p_B > 0.05 else "No"},
    {"Dataset": "B (n=45)",  "Prueba": "Kolmogorov-Smirnov",
     "Estadístico": round(ks_stat_B, 6), "p-valor": round(ks_p_B, 6),
     "¿Normal? (α=0.05)": "Sí" if ks_p_B > 0.05 else "No"},
])
print("\nTabla 3: Resultados de pruebas de normalidad sobre residuos")
print(tabla_norm.to_string(index=False))


# =============================================================================
# GRÁFICO COMPARATIVO FINAL: sobreestimación / subestimación (Fase IV apoyo)
# =============================================================================
def plot_real_vs_pred(metricas, nombre_ds, tag):
    """Real vs predicho para análisis de sobreestimación / subestimación."""
    y_real = metricas["y_real"]
    y_pred = metricas["y_pred"]
    residuos = metricas["residuos"]   # positivo = subestimación, negativo = sobreestimación

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Real vs. Predicho y Patrón de Errores – Dataset {nombre_ds}",
                 fontweight="bold")

    # Panel izquierdo: real vs predicho
    minv = min(y_real.min(), y_pred.min())
    maxv = max(y_real.max(), y_pred.max())
    axes[0].scatter(y_pred, y_real, alpha=0.5, color="steelblue",
                    edgecolors="none", s=18)
    axes[0].plot([minv, maxv], [minv, maxv], "r--", linewidth=1.5,
                 label="Predicción perfecta")
    axes[0].set_xlabel("Effort Predicho (ŷ)")
    axes[0].set_ylabel("Effort Real (y)")
    axes[0].set_title("Real vs. Predicho")
    axes[0].legend()

    # Panel derecho: residuos con color según signo
    colores = np.where(residuos >= 0, "steelblue", "darkorange")
    axes[1].scatter(y_pred, residuos, c=colores, alpha=0.5, edgecolors="none", s=18)
    axes[1].axhline(0, color="black", linewidth=1.2, linestyle="--")
    axes[1].set_xlabel("Effort Predicho (ŷ)")
    axes[1].set_ylabel("Residuo (y − ŷ)")
    axes[1].set_title("Subestimación (azul) / Sobreestimación (naranja)")
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor="steelblue", label="Subestimación (y > ŷ)"),
                    Patch(facecolor="darkorange", label="Sobreestimación (y < ŷ)")]
    axes[1].legend(handles=legend_elems, fontsize=9)

    plt.tight_layout()
    ruta = os.path.join(OUTPUT_DIR, f"real_vs_pred_{tag}.png")
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {ruta}")

plot_real_vs_pred(metricas_A, "A (n=429)", "A")
plot_real_vs_pred(metricas_B, "B (n=45)",  "B")


# =============================================================================
# RESUMEN FINAL EN CONSOLA
# =============================================================================
print("\n" + "=" * 60)
print("RESUMEN FINAL")
print("=" * 60)
print(f"\n  Dataset A — Ecuación: Effort = {metricas_A['b1']:.4f}·FSM + ({metricas_A['b0']:.4f})")
print(f"              R² = {metricas_A['r2']:.4f}  |  RMSE = {metricas_A['rmse']:.2f}  |  MAE = {metricas_A['mae']:.2f}")
print(f"\n  Dataset B — Ecuación: Effort = {metricas_B['b1']:.4f}·FSM + ({metricas_B['b0']:.4f})")
print(f"              R² = {metricas_B['r2']:.4f}  |  RMSE = {metricas_B['rmse']:.2f}  |  MAE = {metricas_B['mae']:.2f}")
print(f"\n  Prueba comparativa ({metodo_cmp}): estadístico = {stat_cmp:.4f}, p = {p_cmp:.6f}")
print(f"\n  Imágenes generadas en la carpeta: ./{OUTPUT_DIR}/")
print("=" * 60)
print("Script finalizado correctamente.")
