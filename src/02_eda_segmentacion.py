"""Fase 2 - Analisis exploratorio (EDA) y segmentacion.

Parte del panel de la Fase 1 y responde las preguntas que definen el torneo de modelos:
cuanto volumen concentran las series vigentes y modelables, que tan estacional es la
demanda, que tan sensibles son los umbrales de inactividad e historia minima, y cuantos
atipicos hay. Deja una tabla de segmentos por serie para la Fase 3.

Entradas
    data/processed/panel_mensual.parquet
    data/processed/perfil_series.parquet

Salidas
    data/processed/segmentos.parquet      perfil + volumen reciente, ABC, estacionalidad, atipicos
    outputs/figures/02_*.png              figuras para el documento
    outputs/reports/02_eda.md             resumen (solo agregados, sin filas)

Uso
    python src/02_eda_segmentacion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from statsmodels.tsa.seasonal import STL  # noqa: E402
from statsmodels.tsa.stattools import acf  # noqa: E402

# --------------------------------------------------------------------------------------
# Configuracion
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "outputs" / "reports"
FIGURES_DIR = ROOT / "outputs" / "figures"

# Debe coincidir con MESES_MINIMOS_MODELADO de la Fase 1.
HISTORIA_MINIMA = 24

# Ventana para medir el volumen reciente (base del ABC y de las coberturas).
MESES_VOLUMEN = 12

# Umbrales alternativos para el analisis de sensibilidad.
UMBRALES_INACTIVIDAD = [3, 6, 9, 12]
UMBRALES_HISTORIA = [12, 18, 24]

# Cortes del ABC sobre el volumen acumulado.
CORTES_ABC = (0.80, 0.95)

PERIODO_ESTACIONAL = 12

# Atipico: venta mensual positiva por encima de mediana + Z * MAD (escalado a desviacion).
Z_ATIPICO = 5.0

SERIE = ["centro", "sku"]
CLASES = ["suave", "erratica", "intermitente", "irregular"]


# --------------------------------------------------------------------------------------
# Calculos
# --------------------------------------------------------------------------------------
def volumen_reciente(panel: pd.DataFrame, meses: int) -> pd.Series:
    fin = panel["periodo"].max()
    desde = fin - pd.DateOffset(months=meses - 1)
    ventana = panel[panel["periodo"] >= desde]
    return ventana.groupby(SERIE)["demanda"].sum().rename("volumen_reciente")


def clasificar_abc(volumen: pd.Series) -> pd.Series:
    """ABC por centro: A hasta el 80 % del volumen acumulado, B hasta el 95 %, C el resto."""

    def por_centro(s: pd.Series) -> pd.Series:
        s = s.sort_values(ascending=False)
        total = s.sum()
        if total <= 0:
            return pd.Series("C", index=s.index)
        inicio = (s.cumsum() - s) / total  # volumen acumulado antes de la serie
        etiquetas = np.select([inicio < CORTES_ABC[0], inicio < CORTES_ABC[1]], ["A", "B"], "C")
        return pd.Series(etiquetas, index=s.index)

    return volumen.groupby(level="centro", group_keys=False).apply(por_centro).rename("abc")


def fuerza_stl(y: pd.Series):
    """Fuerza de tendencia y estacionalidad (Wang, Smith y Hyndman, 2006) con STL robusto.

    Los meses faltantes se interpolan solo para este diagnostico.
    """
    y = y.sort_index().astype(float).interpolate(limit_direction="both")
    if len(y) < 2 * PERIODO_ESTACIONAL or y.std() == 0:
        return np.nan, np.nan, None
    res = STL(y.to_numpy(), period=PERIODO_ESTACIONAL, robust=True).fit()
    var_resid = np.var(res.resid)

    def fuerza(componente) -> float:
        denom = np.var(componente + res.resid)
        return float(max(0.0, 1 - var_resid / denom)) if denom > 0 else np.nan

    return fuerza(res.trend), fuerza(res.seasonal), res


def contar_atipicos(y: pd.Series) -> int:
    positivos = y[y > 0]
    if len(positivos) < 3:
        return 0
    mediana = positivos.median()
    mad = 1.4826 * (positivos - mediana).abs().median()
    if mad == 0:
        return 0
    return int((positivos > mediana + Z_ATIPICO * mad).sum())


def diagnostico_series(panel: pd.DataFrame, claves: pd.DataFrame) -> pd.DataFrame:
    """Tendencia, estacionalidad y atipicos de cada serie modelable, sobre su periodo activo."""
    sub = panel.merge(claves, on=SERIE)
    sub = sub[sub["activo"]]
    filas = []
    for (centro, sku), g in sub.groupby(SERIE):
        y = g.set_index("periodo")["demanda"]
        ft, fs, _ = fuerza_stl(y)
        filas.append(
            {"centro": centro, "sku": sku, "fuerza_tendencia": ft, "fuerza_estacional": fs,
             "n_atipicos": contar_atipicos(y.dropna())}
        )
    columnas = SERIE + ["fuerza_tendencia", "fuerza_estacional", "n_atipicos"]
    return pd.DataFrame(filas, columns=columnas)


def demanda_agregada(panel: pd.DataFrame) -> pd.DataFrame:
    """Demanda mensual total por centro (NaN en los meses faltantes de un centro)."""
    return panel.groupby(["periodo", "centro"])["demanda"].sum(min_count=1).unstack("centro")


# --------------------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------------------
def guardar(fig, nombre: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / nombre, dpi=150)
    plt.close(fig)


def figura_demanda(agregada: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for centro in agregada.columns:
        ax.plot(agregada.index, agregada[centro], marker="o", ms=3, label=centro)
    ax.set_title("Demanda mensual total por sede (unidades)")
    ax.set_ylabel("Unidades")
    ax.legend()
    ax.grid(alpha=0.3)
    guardar(fig, "02_demanda_mensual_por_sede.png")


def figura_skus_activos(panel: pd.DataFrame) -> None:
    con_venta = panel[panel["demanda"] > 0].groupby(["periodo", "centro"]).size().unstack("centro")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for centro in con_venta.columns:
        ax.plot(con_venta.index, con_venta[centro], marker="o", ms=3, label=centro)
    ax.set_title("SKUs con venta por mes y sede")
    ax.set_ylabel("SKUs")
    ax.legend()
    ax.grid(alpha=0.3)
    guardar(fig, "02_skus_con_venta_por_mes.png")


def figura_syntetos_boylan(seg: pd.DataFrame) -> None:
    vig = seg[(seg["estado_ciclo"] == "vigente") & seg["adi"].notna() & seg["cv2"].notna()]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for modelable, color, etiqueta in [(False, "#9aa5b1", "vigente, historia corta"),
                                       (True, "#1f6feb", f"modelable (>= {HISTORIA_MINIMA} meses)")]:
        d = vig[vig["modelable"] == modelable]
        ax.scatter(d["adi"], d["cv2"], s=12, alpha=0.6, color=color, label=etiqueta)
    ax.axvline(1.32, color="k", lw=0.8, ls="--")
    ax.axhline(0.49, color="k", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("ADI (intervalo medio entre ventas, meses)")
    ax.set_ylabel("CV² de las ventas positivas")
    ax.set_title("Clasificación de Syntetos-Boylan (series vigentes)")
    ax.legend(loc="upper right")
    guardar(fig, "02_syntetos_boylan.png")


def figura_pareto(seg: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    vig = seg[seg["estado_ciclo"] == "vigente"]
    for centro, g in vig.groupby("centro"):
        v = g["volumen_reciente"].sort_values(ascending=False).to_numpy()
        if v.sum() <= 0:
            continue
        x = np.arange(1, len(v) + 1) / len(v)
        ax.plot(100 * x, 100 * np.cumsum(v) / v.sum(), label=centro)
    for corte in CORTES_ABC:
        ax.axhline(100 * corte, color="k", lw=0.8, ls="--")
    ax.set_xlabel("% de series vigentes (ordenadas por volumen)")
    ax.set_ylabel(f"% acumulado del volumen de los últimos {MESES_VOLUMEN} meses")
    ax.set_title("Curva de Pareto (ABC) por sede")
    ax.legend()
    ax.grid(alpha=0.3)
    guardar(fig, "02_pareto_abc.png")


def figura_stl_total(agregada: pd.DataFrame) -> None:
    total = agregada.interpolate(limit_direction="both").sum(axis=1)
    _, _, res = fuerza_stl(total)
    if res is None:
        return
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for ax, (nombre, valores) in zip(
        axes, [("Observada", total.to_numpy()), ("Tendencia", res.trend),
               ("Estacional", res.seasonal), ("Residuo", res.resid)]
    ):
        ax.plot(total.index, valores)
        ax.set_ylabel(nombre)
        ax.grid(alpha=0.3)
    axes[0].set_title("Descomposición STL de la demanda total (3 sedes)")
    guardar(fig, "02_stl_demanda_total.png")


def figura_historia(seg: pd.DataFrame) -> None:
    vig = seg[seg["estado_ciclo"] == "vigente"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(vig["n_meses"], bins=range(1, 32), color="#1f6feb", alpha=0.8)
    for h in UMBRALES_HISTORIA:
        ax.axvline(h - 0.5, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Meses de historia activa")
    ax.set_ylabel("Series vigentes")
    ax.set_title("Historia disponible de las series vigentes")
    guardar(fig, "02_historia_series_vigentes.png")


# --------------------------------------------------------------------------------------
# Reporte
# --------------------------------------------------------------------------------------
def bloque(titulo: str, contenido) -> str:
    if isinstance(contenido, (pd.DataFrame, pd.Series)):
        contenido = contenido.to_string()
    return f"## {titulo}\n\n```\n{contenido}\n```\n"


def participacion(seg: pd.DataFrame, filas) -> pd.DataFrame:
    """Series y % del volumen reciente por grupo."""
    total = seg["volumen_reciente"].sum()
    t = seg.groupby(filas).agg(series=("sku", "size"), volumen=("volumen_reciente", "sum"))
    t["% volumen"] = (100 * t["volumen"] / total).round(1) if total > 0 else np.nan
    return t.drop(columns="volumen")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(PROCESSED_DIR / "panel_mensual.parquet")
    perfil = pd.read_parquet(PROCESSED_DIR / "perfil_series.parquet")

    seg = perfil.set_index(SERIE)
    seg["modelable"] = (seg["estado_ciclo"] == "vigente") & (seg["n_meses"] >= HISTORIA_MINIMA)
    seg = seg.join(volumen_reciente(panel, MESES_VOLUMEN))
    seg["volumen_reciente"] = seg["volumen_reciente"].fillna(0)
    seg["abc"] = clasificar_abc(seg.loc[seg["estado_ciclo"] == "vigente", "volumen_reciente"])
    seg = seg.reset_index()

    diag = diagnostico_series(panel, seg.loc[seg["modelable"], SERIE])
    seg = seg.merge(diag, on=SERIE, how="left")
    seg.to_parquet(PROCESSED_DIR / "segmentos.parquet", index=False)

    # ---- Agregados por sede --------------------------------------------------------
    agregada = demanda_agregada(panel)
    estacionalidad_sede = {}
    for centro in list(agregada.columns) + ["TOTAL"]:
        y = agregada.interpolate(limit_direction="both").sum(axis=1) if centro == "TOTAL" else agregada[centro]
        ft, fs, _ = fuerza_stl(y)
        y_lleno = y.interpolate(limit_direction="both")
        rho12 = acf(y_lleno, nlags=PERIODO_ESTACIONAL, fft=False)[PERIODO_ESTACIONAL] if len(y_lleno) > PERIODO_ESTACIONAL else np.nan
        anual = y.groupby(y.index.year).sum(min_count=1)
        meses_por_anio = y.groupby(y.index.year).size()
        estacionalidad_sede[centro] = {
            "fuerza_tendencia": round(ft, 2), "fuerza_estacional": round(fs, 2),
            "acf_lag12": round(float(rho12), 2),
            **{
                f"total_{a}" + ("" if meses_por_anio[a] == 12 else f"_{meses_por_anio[a]}m"): round(float(v))
                for a, v in anual.items()
            },
        }
    estacionalidad_sede = pd.DataFrame(estacionalidad_sede).T

    # ---- Sensibilidad de umbrales --------------------------------------------------
    total_reciente = seg["volumen_reciente"].sum()
    sens_inactividad = pd.DataFrame(
        {
            f"< {t} meses sin venta": {
                "series vigentes": int((seg["meses_sin_venta_al_cierre"] < t).sum()),
                "% volumen reciente": round(100 * seg.loc[seg["meses_sin_venta_al_cierre"] < t, "volumen_reciente"].sum() / total_reciente, 1),
            }
            for t in UMBRALES_INACTIVIDAD
        }
    ).T
    vig = seg[seg["estado_ciclo"] == "vigente"]
    sens_historia = pd.DataFrame(
        {
            f">= {h} meses": {
                "series": int((vig["n_meses"] >= h).sum()),
                "% volumen reciente": round(100 * vig.loc[vig["n_meses"] >= h, "volumen_reciente"].sum() / total_reciente, 1),
            }
            for h in UMBRALES_HISTORIA
        }
    ).T

    # ---- Diagnostico de las series modelables ---------------------------------------
    mod = seg[seg["modelable"]]
    diag_clase = mod.groupby("clase_demanda").agg(
        series=("sku", "size"),
        fuerza_tendencia_mediana=("fuerza_tendencia", "median"),
        fuerza_estacional_mediana=("fuerza_estacional", "median"),
        series_estacionales_fs_mayor_064=("fuerza_estacional", lambda s: int((s > 0.64).sum())),
        series_con_atipicos=("n_atipicos", lambda s: int((s > 0).sum())),
        atipicos_totales=("n_atipicos", "sum"),
    ).round(2)

    figura_demanda(agregada)
    figura_skus_activos(panel)
    figura_syntetos_boylan(seg)
    figura_pareto(seg)
    figura_stl_total(agregada)
    figura_historia(seg)

    secciones = [
        "# Fase 2 - EDA y segmentacion\n",
        bloque(f"Volumen de los ultimos {MESES_VOLUMEN} meses por estado y modelabilidad", participacion(seg, ["estado_ciclo", "modelable"])),
        bloque("Volumen por sede, estado y modelabilidad", participacion(seg, ["centro", "estado_ciclo", "modelable"])),
        bloque("Series vigentes por ABC y clase de demanda", pd.crosstab(vig["abc"], vig["clase_demanda"], margins=True)),
        bloque("Series vigentes por ABC y modelabilidad (con % de volumen)", participacion(vig, ["abc", "modelable"])),
        bloque("Tendencia y estacionalidad de la demanda agregada por sede (STL, ACF)", estacionalidad_sede),
        bloque("Sensibilidad: umbral de inactividad", sens_inactividad),
        bloque("Sensibilidad: historia minima (series vigentes)", sens_historia),
        bloque("Series modelables: tendencia, estacionalidad y atipicos por clase", diag_clase),
        bloque("Figuras generadas", "\n".join(sorted(p.name for p in FIGURES_DIR.glob("02_*.png")))),
    ]
    reporte = "\n".join(secciones)
    (REPORTS_DIR / "02_eda.md").write_text(reporte, encoding="utf-8")
    print(reporte)
    print(f"Reporte guardado en {REPORTS_DIR / '02_eda.md'}")


if __name__ == "__main__":
    main()
