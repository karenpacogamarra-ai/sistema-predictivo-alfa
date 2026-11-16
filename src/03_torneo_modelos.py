"""Fase 3 - Torneo de modelos.

Compite un conjunto de modelos clasicos y de demanda intermitente sobre las series
modelables (vigentes y con historia suficiente), con validacion cruzada de origen movil.
Elige un ganador por serie y compara los modelos con la prueba de Friedman.

Entradas
    data/processed/panel_mensual.parquet
    data/processed/segmentos.parquet

Salidas
    data/processed/cv_predicciones.parquet  pronosticos de cada ventana de validacion
    data/processed/cv_metricas.parquet      metricas por serie y modelo
    data/processed/ganadores.parquet        modelo ganador por serie
    outputs/reports/03_torneo.md            resumen (solo agregados, sin filas)

Uso
    python src/03_torneo_modelos.py
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("NIXTLA_ID_AS_COL", "1")
# Divisiones por cero internas de AutoETS en series cortas; los procesos en paralelo leen PYTHONWARNINGS.
os.environ.setdefault("PYTHONWARNINGS", "ignore::RuntimeWarning")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsforecast")

import numpy as np
import pandas as pd
from scipy import stats
from statsforecast import StatsForecast
from statsforecast.models import (
    ADIDA,
    IMAPA,
    TSB,
    AutoARIMA,
    AutoETS,
    CrostonClassic,
    CrostonSBA,
    HistoricAverage,
    Naive,
    SeasonalNaive,
    SimpleExponentialSmoothingOptimized,
    WindowAverage,
)

# --------------------------------------------------------------------------------------
# Configuracion
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "outputs" / "reports"

# Horizonte de pronostico en meses. Con reposicion quincenal y lead time de 1-3 dias el
# intervalo de proteccion es de 16-18 dias: basta el pronostico del mes siguiente.
HORIZONTE = 1

# Ventanas de validacion (origenes moviles de 1 mes): se evaluan los ultimos 12 meses
# (2025-07 a 2026-06).
N_VENTANAS = 12

# Evaluacion honesta de la seleccion: el ganador se elige con las primeras ventanas y se
# mide en las restantes, que no participaron en la eleccion (8 + 4).
N_VENTANAS_SELECCION = 8

PERIODO_ESTACIONAL = 12

# El catalogo cambia cada 2 meses: estacional ingenuo de periodo 2 como hipotesis de campana.
PERIODO_CATALOGO = 2

# Prophet esta comprometido en el objetivo especifico 2. Es lento (un ajuste por serie y
# ventana): ponlo en False para corridas de prueba.
INCLUIR_PROPHET = True

# Terminos de Fourier de la estacionalidad anual de Prophet. El valor por defecto (10) son
# 20 parametros para 24-30 datos mensuales: sobreajusta y cada ajuste tarda ~25 s.
TERMINOS_FOURIER_PROPHET = 3

SERIE = ["centro", "sku"]

# Valores criticos q_0.05 de Nemenyi (Demsar, 2006) segun el numero de modelos k.
Q_NEMENYI_005 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
                 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391,
                 16: 3.426, 17: 3.459, 18: 3.489, 19: 3.517, 20: 3.544}


def modelos():
    return [
        Naive(alias="Naive"),
        SeasonalNaive(season_length=PERIODO_ESTACIONAL, alias="SeasonalNaive"),
        SeasonalNaive(season_length=PERIODO_CATALOGO, alias="Bimestral"),
        HistoricAverage(alias="Media"),
        # Metodo actual de la empresa: promedio de ventas de los ultimos 6 meses.
        WindowAverage(window_size=6, alias="MetodoActual6M"),
        SimpleExponentialSmoothingOptimized(alias="SES"),
        AutoETS(season_length=PERIODO_ESTACIONAL, alias="ETS"),
        AutoARIMA(season_length=PERIODO_ESTACIONAL, alias="ARIMA"),
        CrostonClassic(alias="Croston"),
        CrostonSBA(alias="SBA"),
        TSB(alpha_d=0.2, alpha_p=0.2, alias="TSB"),
        ADIDA(alias="ADIDA"),
        IMAPA(alias="IMAPA"),
    ]


# --------------------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------------------
def preparar_series(panel: pd.DataFrame, seg: pd.DataFrame) -> pd.DataFrame:
    """Formato largo de statsforecast (unique_id, ds, y) para las series modelables.

    Solo el periodo activo de cada serie. Los meses faltantes de un centro (Cochabamba
    2024-11) se imputan por interpolacion lineal entre el mes anterior y el siguiente.
    """
    claves = seg.loc[seg["modelable"], SERIE]
    df = panel.merge(claves, on=SERIE)
    df = df[df["activo"]].sort_values(SERIE + ["periodo"]).copy()
    df["imputado"] = df["demanda"].isna()
    df["demanda"] = (
        df.groupby(SERIE)["demanda"].transform(lambda s: s.interpolate(limit_direction="both"))
    )
    df["unique_id"] = df["centro"] + "|" + df["sku"]
    return df.rename(columns={"periodo": "ds", "demanda": "y"})[
        ["unique_id", "ds", "y", "imputado"]
    ].reset_index(drop=True)


def validar_cruzado(series: pd.DataFrame) -> pd.DataFrame:
    sf = StatsForecast(models=modelos(), freq="MS", n_jobs=-1, fallback_model=Naive())
    cv = sf.cross_validation(
        df=series[["unique_id", "ds", "y"]], h=HORIZONTE, step_size=1, n_windows=N_VENTANAS
    )
    if "unique_id" not in cv.columns:
        cv = cv.reset_index()
    if INCLUIR_PROPHET:
        cv = cv.merge(prophet_cv(series, cv), on=["unique_id", "ds", "cutoff"], how="left")
    return cv


def prophet_cv(series: pd.DataFrame, cv: pd.DataFrame) -> pd.DataFrame:
    import logging

    from prophet import Prophet

    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    filas = []
    cortes = cv[["unique_id", "cutoff"]].drop_duplicates()
    por_serie = dict(tuple(series.groupby("unique_id")))
    for uid, cutoff in cortes.itertuples(index=False):
        hist = por_serie[uid]
        train = hist[hist["ds"] <= cutoff][["ds", "y"]]
        m = Prophet(
            yearly_seasonality=TERMINOS_FOURIER_PROPHET,
            weekly_seasonality=False,
            daily_seasonality=False,
        )
        m.fit(train)
        futuro = pd.DataFrame({"ds": pd.date_range(cutoff, periods=HORIZONTE + 1, freq="MS")[1:]})
        pred = m.predict(futuro)[["ds", "yhat"]].rename(columns={"yhat": "Prophet"})
        pred["unique_id"] = uid
        pred["cutoff"] = cutoff
        filas.append(pred)
    return pd.concat(filas, ignore_index=True)


# --------------------------------------------------------------------------------------
# Metricas
# --------------------------------------------------------------------------------------
def escalas(series: pd.DataFrame, cv: pd.DataFrame) -> pd.DataFrame:
    """Escala de MASE y RMSSE: error del naive a 1 paso en el entrenamiento de la primera ventana."""
    primer_corte = cv.groupby("unique_id")["cutoff"].min().rename("primer_corte")
    train = series.merge(primer_corte, on="unique_id")
    train = train[train["ds"] <= train["primer_corte"]].sort_values(["unique_id", "ds"])
    dif = train.groupby("unique_id")["y"].diff()
    return pd.DataFrame(
        {"escala_mae": dif.abs().groupby(train["unique_id"]).mean(),
         "escala_mse": (dif ** 2).groupby(train["unique_id"]).mean()}
    )


def calcular_metricas(cv: pd.DataFrame, series: pd.DataFrame) -> pd.DataFrame:
    largo = cv.melt(id_vars=["unique_id", "ds", "cutoff", "y"], var_name="modelo", value_name="pred")
    largo["pred"] = largo["pred"].clip(lower=0)  # la demanda no puede ser negativa
    largo["error"] = largo["y"] - largo["pred"]
    m = largo.groupby(["unique_id", "modelo"]).agg(
        mae=("error", lambda e: e.abs().mean()),
        mse=("error", lambda e: (e ** 2).mean()),
        sesgo=("error", "mean"),
    )
    m = m.join(escalas(series, cv), on="unique_id")
    m["mase"] = m["mae"] / m["escala_mae"].replace(0, np.nan)
    m["rmsse"] = np.sqrt(m["mse"] / m["escala_mse"].replace(0, np.nan))
    m["sesgo_escalado"] = m["sesgo"] / m["escala_mae"].replace(0, np.nan)
    return m.reset_index()


def elegir_ganadores(metricas: pd.DataFrame) -> pd.DataFrame:
    """Menor RMSSE por serie; empate por menor sesgo absoluto.

    Se usa RMSSE y no MAE/MASE porque en series con muchos ceros los criterios absolutos
    premian pronosticar cero, que deja sin stock; RMSSE premia acertar la media.
    """
    orden = metricas.assign(abs_sesgo=metricas["sesgo"].abs()).sort_values(
        ["unique_id", "rmsse", "abs_sesgo"], na_position="last"
    )
    return orden.groupby("unique_id").head(1).drop(columns="abs_sesgo").reset_index(drop=True)


def evaluar_seleccion(cv: pd.DataFrame, series: pd.DataFrame, clases: pd.Series):
    """Compara estrategias de seleccion en ventanas que no se usaron para elegir.

    - ganador por serie: el de menor RMSSE de cada serie en las ventanas de seleccion.
    - modelo por clase: el de mejor rango promedio dentro de cada clase de demanda.
    - modelos unicos: el mismo modelo para todas las series.
    """
    cortes = np.sort(cv["cutoff"].unique())
    sel, ev = cortes[:N_VENTANAS_SELECCION], cortes[N_VENTANAS_SELECCION:]
    m_sel = calcular_metricas(cv[cv["cutoff"].isin(sel)], series)
    m_ev = calcular_metricas(cv[cv["cutoff"].isin(ev)], series)

    gan_serie = elegir_ganadores(m_sel).set_index("unique_id")["modelo"]
    m_sel = m_sel.join(clases, on="unique_id")
    m_sel["rango"] = m_sel.groupby("unique_id")["rmsse"].rank()
    rango_clase = m_sel.groupby(["clase_demanda", "modelo"])["rango"].mean()
    gan_clase = rango_clase.groupby(level=0).idxmin().map(lambda t: t[1])

    tabla = m_ev.pivot(index="unique_id", columns="modelo", values="rmsse")
    modelo_clase = clases.reindex(tabla.index).map(gan_clase)

    def tomar(asignacion: pd.Series) -> pd.Series:
        return pd.Series([tabla.at[u, m] for u, m in asignacion.items()], index=asignacion.index)

    estrategias = {
        "ganador por serie": tomar(gan_serie.reindex(tabla.index)),
        "modelo por clase de demanda": tomar(modelo_clase),
        **{
            f"modelo unico: {m}": tabla[m]
            for m in ["MetodoActual6M", "SeasonalNaive", "Naive", "Media", "SBA", "ETS", "Prophet"]
            if m in tabla
        },
    }
    resumen = pd.DataFrame(
        {nombre: {"rmsse_mediana": s.median(), "rmsse_media": s.mean()} for nombre, s in estrategias.items()}
    ).T.sort_values("rmsse_mediana").round(3)
    periodo = f"{pd.Timestamp(ev.min()):%Y-%m} a {pd.Timestamp(ev.max()):%Y-%m}"
    return resumen, gan_clase.rename("modelo elegido"), periodo


def prueba_friedman(metricas: pd.DataFrame):
    """Friedman sobre el RMSSE de todas las series y rangos promedio con la diferencia critica de Nemenyi."""
    tabla = metricas.pivot(index="unique_id", columns="modelo", values="rmsse").dropna()
    k, n = tabla.shape[1], tabla.shape[0]
    estadistico, p = stats.friedmanchisquare(*[tabla[c] for c in tabla.columns])
    rangos = tabla.rank(axis=1).mean().sort_values()
    q = Q_NEMENYI_005.get(k)
    cd = q * np.sqrt(k * (k + 1) / (6 * n)) if q else np.nan
    return rangos, estadistico, p, cd, n


# --------------------------------------------------------------------------------------
# Reporte
# --------------------------------------------------------------------------------------
def bloque(titulo: str, contenido) -> str:
    if isinstance(contenido, (pd.DataFrame, pd.Series)):
        contenido = contenido.to_string()
    return f"## {titulo}\n\n```\n{contenido}\n```\n"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(PROCESSED_DIR / "panel_mensual.parquet")
    seg = pd.read_parquet(PROCESSED_DIR / "segmentos.parquet")
    seg["unique_id"] = seg["centro"] + "|" + seg["sku"]

    series = preparar_series(panel, seg)
    inicio = time.perf_counter()
    cv = validar_cruzado(series)
    duracion = time.perf_counter() - inicio

    metricas = calcular_metricas(cv, series)
    ganadores = elegir_ganadores(metricas)
    atributos = seg.set_index("unique_id")[["centro", "clase_demanda", "abc"]]
    metricas = metricas.join(atributos, on="unique_id")
    ganadores = ganadores.join(atributos, on="unique_id")

    cv.to_parquet(PROCESSED_DIR / "cv_predicciones.parquet", index=False)
    metricas.to_parquet(PROCESSED_DIR / "cv_metricas.parquet", index=False)
    ganadores.to_parquet(PROCESSED_DIR / "ganadores.parquet", index=False)

    # ---- Agregados para el reporte --------------------------------------------------
    rangos, chi2, p, cd, n_friedman = prueba_friedman(metricas)
    estrategias, gan_clase, periodo_eval = evaluar_seleccion(
        cv, series, seg.set_index("unique_id")["clase_demanda"]
    )
    resumen_modelos = metricas.groupby("modelo").agg(
        rmsse_mediana=("rmsse", "median"),
        mase_mediana=("mase", "median"),
        sesgo_escalado_mediana=("sesgo_escalado", "median"),
    ).join(rangos.rename("rango_promedio")).sort_values("rango_promedio").round(3)
    resumen_modelos["victorias"] = ganadores["modelo"].value_counts().reindex(resumen_modelos.index).fillna(0).astype(int)

    rmsse = metricas.pivot(index="unique_id", columns="modelo", values="rmsse")
    gan_rmsse = ganadores.set_index("unique_id")["rmsse"]
    contra_benchmarks = pd.Series(
        {
            f"% series donde el ganador mejora a {b}": round(100 * (gan_rmsse < rmsse[b]).mean(), 1)
            for b in ["MetodoActual6M", "Naive", "SeasonalNaive", "Media"] if b in rmsse
        }
    )
    generales = pd.Series(
        {
            "series en el torneo": series["unique_id"].nunique(),
            "meses imputados (faltantes de centro)": int(series["imputado"].sum()),
            "horizonte (meses)": HORIZONTE,
            "ventanas de validacion": N_VENTANAS,
            "primer y ultimo origen": f"{cv['cutoff'].min():%Y-%m} a {cv['cutoff'].max():%Y-%m}",
            "modelos": metricas["modelo"].nunique(),
            "Prophet incluido": INCLUIR_PROPHET,
            "duracion de la validacion (s)": round(duracion, 1),
        }
    )
    friedman = pd.Series(
        {
            "series con RMSSE en todos los modelos": n_friedman,
            "chi2 de Friedman": round(chi2, 2),
            "p-valor": f"{p:.2e}",
            "diferencia critica de Nemenyi (alfa 0.05)": round(cd, 3),
        }
    )

    secciones = [
        "# Fase 3 - Torneo de modelos\n",
        bloque("Configuracion", generales),
        bloque("Resumen por modelo (ordenado por rango promedio de RMSSE)", resumen_modelos),
        bloque("Prueba de Friedman sobre RMSSE", friedman),
        bloque("Ganador frente a los modelos de referencia", contra_benchmarks),
        bloque(
            f"Evaluacion honesta de la seleccion: elegido con las primeras {N_VENTANAS_SELECCION} "
            f"ventanas, medido en los origenes {periodo_eval}",
            estrategias,
        ),
        bloque("Modelo elegido por clase de demanda (ventanas de seleccion)", gan_clase),
        bloque("Victorias por modelo y clase de demanda", pd.crosstab(ganadores["modelo"], ganadores["clase_demanda"], margins=True)),
        bloque("Victorias por modelo y ABC", pd.crosstab(ganadores["modelo"], ganadores["abc"], margins=True)),
        bloque("RMSSE mediano por modelo y clase de demanda", metricas.pivot_table(index="modelo", columns="clase_demanda", values="rmsse", aggfunc="median").round(3)),
        bloque("Victorias por modelo y sede", pd.crosstab(ganadores["modelo"], ganadores["centro"], margins=True)),
    ]
    reporte = "\n".join(secciones)
    (REPORTS_DIR / "03_torneo.md").write_text(reporte, encoding="utf-8")
    print(reporte)
    print(f"Reporte guardado en {REPORTS_DIR / '03_torneo.md'}")


if __name__ == "__main__":
    main()
