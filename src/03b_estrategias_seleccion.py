"""Fase 3b - Estrategias de seleccion de modelos.

Reutiliza los pronosticos de validacion de la Fase 3 (no reentrena nada) y compara, en
ventanas que no participaron en la eleccion, varias formas de asignar un modelo a cada
combinacion producto-sucursal:

    1. ganador por serie      el modelo de menor RMSSE de cada serie (lo planteado en el OE2)
    2. hibrida por clase      el ganador de la serie si mejora al menos MARGEN_HIBRIDA al
                              modelo de su clase; si no, el modelo de la clase
    3. modelo por clase       el de mejor rango promedio en cada clase de demanda
    4. combinacion            promedio de los modelos del grupo lider de la Fase 3
    5. hibrida por combinacion como la 2, con la combinacion como respaldo
    6. metodo actual          venta del mes anterior (Naive): asi repone hoy la empresa, segun
                              la gerencia (la planilla muestra lo vendido el mes pasado)
    7. promedio 6 meses       promedio de los ultimos 6 meses, referencia simple mas exigente

Cada estrategia se contrasta con el metodo actual y con el promedio de 6 meses mediante la
prueba de Wilcoxon de rangos con signo (unilateral, pareada por serie). Al final guarda la asignacion de cada estrategia
elegida con todas las ventanas, que es la que usara la Fase 4.

Entradas
    data/processed/panel_mensual.parquet
    data/processed/segmentos.parquet
    data/processed/cv_predicciones.parquet

Salidas
    data/processed/asignacion_modelos.parquet   modelo por serie para cada estrategia
    outputs/reports/03b_estrategias.md          resumen (solo agregados, sin filas)

Uso
    python src/03b_estrategias_seleccion.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "outputs" / "reports"

# Grupo lider de la Fase 3 (sin diferencia significativa segun Nemenyi), sin el metodo actual.
MODELOS_COMBINACION = ["SBA", "Media", "IMAPA", "Croston", "ADIDA", "TSB"]

# Mejora relativa minima de RMSSE para preferir el modelo propio de la serie al respaldo.
MARGEN_HIBRIDA = 0.10

# Con lead time de 1-3 dias y reposicion quincenal, el pedido cubre unos 18 dias: el
# horizonte que importa para el inventario es el primer mes.
HORIZONTE_OPERATIVO = 1

# Metodo actual de la empresa: pedir lo vendido el mes anterior (confirmado por la gerencia).
REFERENCIA = "Naive"
# Segunda referencia: promedio de los ultimos 6 meses (supuesto inicial del proyecto).
REFERENCIA_6M = "MetodoActual6M"
COMBINACIONES = ["Combinacion", "Mediana"]
ID_COLS = ["unique_id", "ds", "cutoff", "y"]


def cargar_fase3():
    """El script de la Fase 3 empieza con un digito, asi que se importa por ruta."""
    ruta = Path(__file__).resolve().parent / "03_torneo_modelos.py"
    spec = importlib.util.spec_from_file_location("fase3", ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


f3 = cargar_fase3()


# --------------------------------------------------------------------------------------
# Estrategias
# --------------------------------------------------------------------------------------
def agregar_combinaciones(cv: pd.DataFrame) -> pd.DataFrame:
    individuales = [c for c in cv.columns if c not in ID_COLS]
    cv = cv.copy()
    lideres = [m for m in MODELOS_COMBINACION if m in cv.columns]
    cv["Combinacion"] = cv[lideres].clip(lower=0).mean(axis=1)
    cv["Mediana"] = cv[individuales].clip(lower=0).median(axis=1)
    return cv


def solo_horizonte(cv: pd.DataFrame, h: int) -> pd.DataFrame:
    """Filas de la validacion cuyo mes pronosticado esta h meses despues del origen."""
    meses = (cv["ds"].dt.year - cv["cutoff"].dt.year) * 12 + (cv["ds"].dt.month - cv["cutoff"].dt.month)
    return cv[meses == h]


def tomar(tabla: pd.DataFrame, asignacion: pd.Series) -> pd.Series:
    """Valor de `tabla` en la columna asignada a cada serie."""
    asignacion = asignacion.reindex(tabla.index)
    cols = tabla.columns.get_indexer(asignacion)
    valores = tabla.to_numpy()[np.arange(len(tabla)), np.clip(cols, 0, None)]
    return pd.Series(np.where(cols >= 0, valores, np.nan), index=tabla.index)


def asignar(metricas: pd.DataFrame, clases: pd.Series, individuales: list[str]) -> dict[str, pd.Series]:
    """Modelo asignado a cada serie por cada estrategia, segun las metricas dadas."""
    tabla = metricas.pivot(index="unique_id", columns="modelo", values="rmsse")
    ind = tabla[individuales]
    clase = clases.reindex(ind.index)

    por_serie = ind.fillna(np.inf).idxmin(axis=1)
    rango_clase = ind.rank(axis=1).groupby(clase).mean()
    por_clase = clase.map(rango_clase.idxmin(axis=1))

    def hibrida(respaldo: pd.Series) -> pd.Series:
        propio = tomar(tabla, por_serie)
        alternativo = tomar(tabla, respaldo)
        return por_serie.where(propio <= (1 - MARGEN_HIBRIDA) * alternativo, respaldo)

    combinacion = pd.Series("Combinacion", index=ind.index)
    return {
        "ganador por serie": por_serie,
        "hibrida por clase": hibrida(por_clase),
        "modelo por clase": por_clase,
        "combinacion": combinacion,
        "hibrida por combinacion": hibrida(combinacion),
        "metodo actual": pd.Series(REFERENCIA, index=ind.index),
        "promedio 6 meses": pd.Series(REFERENCIA_6M, index=ind.index),
    }


def evaluar(asignaciones: dict[str, pd.Series], m_ev: pd.DataFrame) -> pd.DataFrame:
    rmsse = m_ev.pivot(index="unique_id", columns="modelo", values="rmsse")
    sesgo = m_ev.pivot(index="unique_id", columns="modelo", values="sesgo_escalado")
    def contraste(r: pd.Series, base: pd.Series, es_base: bool):
        validas = r.notna() & base.notna()
        p = np.nan
        if not es_base and (r[validas] != base[validas]).any():
            p = stats.wilcoxon(r[validas], base[validas], alternative="less").pvalue
        return 100 * (r[validas] < base[validas]).mean(), p

    filas = {}
    for nombre, asignacion in asignaciones.items():
        r = tomar(rmsse, asignacion)
        pct, p = contraste(r, rmsse[REFERENCIA], nombre == "metodo actual")
        pct6, p6 = contraste(r, rmsse[REFERENCIA_6M], nombre == "promedio 6 meses")
        filas[nombre] = {
            "rmsse_mediana": r.median(),
            "rmsse_media": r.mean(),
            "rmsse_p90": r.quantile(0.9),
            "% series mejor que metodo actual": pct,
            "sesgo_escalado_mediana": tomar(sesgo, asignacion).median(),
            "p Wilcoxon (< metodo actual)": p,
            "% series mejor que promedio 6 meses": pct6,
            "p Wilcoxon (< promedio 6 meses)": p6,
            "modelos distintos usados": asignacion.nunique(),
        }
    tabla = pd.DataFrame(filas).T
    for col in ["p Wilcoxon (< metodo actual)", "p Wilcoxon (< promedio 6 meses)"]:
        tabla[col] = tabla[col].map(lambda v: "" if pd.isna(v) else f"{v:.2e}")
    return tabla.sort_values("rmsse_media")


def bloque(titulo: str, contenido) -> str:
    if isinstance(contenido, (pd.DataFrame, pd.Series)):
        contenido = contenido.to_string(float_format=lambda v: f"{v:.3f}")
    return f"## {titulo}\n\n```\n{contenido}\n```\n"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(PROCESSED_DIR / "panel_mensual.parquet")
    seg = pd.read_parquet(PROCESSED_DIR / "segmentos.parquet")
    seg["unique_id"] = seg["centro"] + "|" + seg["sku"]
    cv_base = pd.read_parquet(PROCESSED_DIR / "cv_predicciones.parquet")

    individuales = [c for c in cv_base.columns if c not in ID_COLS]
    cv = agregar_combinaciones(cv_base)
    series = f3.preparar_series(panel, seg)
    clases = seg.set_index("unique_id")["clase_demanda"]

    cortes = np.sort(cv["cutoff"].unique())
    n_sel = f3.N_VENTANAS_SELECCION
    sel, ev = cortes[:n_sel], cortes[n_sel:]
    m_sel = f3.calcular_metricas(cv[cv["cutoff"].isin(sel)], series)
    m_ev = f3.calcular_metricas(cv[cv["cutoff"].isin(ev)], series)
    m_todas = f3.calcular_metricas(cv, series)

    evaluacion = evaluar(asignar(m_sel, clases, individuales), m_ev)

    # Misma evaluacion, solo con el horizonte operativo (pronostico del mes siguiente).
    cv_h = solo_horizonte(cv, HORIZONTE_OPERATIVO)
    m_sel_h = f3.calcular_metricas(cv_h[cv_h["cutoff"].isin(sel)], series)
    m_ev_h = f3.calcular_metricas(cv_h[cv_h["cutoff"].isin(ev)], series)
    evaluacion_h = evaluar(asignar(m_sel_h, clases, individuales), m_ev_h)

    # Asignacion final con todas las ventanas: la que usa la Fase 4.
    final = pd.DataFrame(asignar(m_todas, clases, individuales))
    final.columns = [c.replace(" ", "_") for c in final.columns]
    final = final.join(seg.set_index("unique_id")[["centro", "sku", "clase_demanda", "abc"]])
    final.reset_index().to_parquet(PROCESSED_DIR / "asignacion_modelos.parquet", index=False)

    rangos, chi2, p, cd, n = f3.prueba_friedman(m_todas)
    friedman = pd.Series(
        {"series": n, "modelos (incluye combinaciones)": len(rangos),
         "chi2 de Friedman": round(chi2, 2), "p-valor": f"{p:.2e}",
         "diferencia critica de Nemenyi (alfa 0.05)": round(cd, 3)}
    )
    hibrida = final["hibrida_por_clase"] == final["ganador_por_serie"]
    uso = pd.Series(
        {
            "series con modelo propio (hibrida por clase)": int(hibrida.sum()),
            "series con el modelo de su clase (hibrida por clase)": int((~hibrida).sum()),
            "series con modelo propio (hibrida por combinacion)": int((final["hibrida_por_combinacion"] == final["ganador_por_serie"]).sum()),
        }
    )
    periodo = f"{pd.Timestamp(ev.min()):%Y-%m} a {pd.Timestamp(ev.max()):%Y-%m}"

    secciones = [
        "# Fase 3b - Estrategias de seleccion\n",
        bloque(
            f"Evaluacion honesta: elegido con {n_sel} ventanas, medido en los origenes {periodo} "
            f"(margen hibrido {MARGEN_HIBRIDA:.0%})",
            evaluacion,
        ),
        *([bloque(
            f"Evaluacion honesta solo con el horizonte operativo (h = {HORIZONTE_OPERATIVO} mes): "
            f"{len(sel)} puntos de seleccion y {len(ev)} de evaluacion por serie",
            evaluacion_h,
        )] if f3.HORIZONTE > HORIZONTE_OPERATIVO else []),
        bloque("Rango promedio de RMSSE con las combinaciones (todas las ventanas)", rangos.round(3)),
        bloque("Prueba de Friedman con las combinaciones", friedman),
        bloque("Uso de modelo propio en la asignacion final (todas las ventanas)", uso),
        bloque("Modelos asignados por la estrategia hibrida por clase", final["hibrida_por_clase"].value_counts()),
        bloque(
            "Modelos asignados por la estrategia hibrida por clase, por sede",
            pd.crosstab(final["hibrida_por_clase"], final["centro"], margins=True),
        ),
    ]
    reporte = "\n".join(secciones)
    (REPORTS_DIR / "03b_estrategias.md").write_text(reporte, encoding="utf-8")
    print(reporte)
    print(f"Reporte guardado en {REPORTS_DIR / '03b_estrategias.md'}")


if __name__ == "__main__":
    main()
