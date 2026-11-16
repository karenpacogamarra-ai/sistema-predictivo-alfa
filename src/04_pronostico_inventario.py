"""Fases 4, 5 y 6 - Pronostico, politica de inventario y validacion frente al metodo actual.

Fase 4 (OE2)  Pronostico del mes siguiente para todas las series vigentes con el modelo de su
              clase de demanda (estrategia elegida en la Fase 3b):
                - series del torneo (>= 24 meses): parametros ajustados a su propia historia;
                - series vigentes con historia corta: mismo modelo de su clase;
                - series inactivas (sin venta en 6 meses): sin pronostico ni reposicion.
Fase 5 (OE3)  Politica de revision periodica (R, S) por producto-sucursal:
                periodo de proteccion  P  = R + L
                stock de seguridad     SS = z * sqrt(P * var_d + d^2 * var_L)
                nivel de reposicion    S  = d * P + SS
                punto de reorden       ROP = d * L + z * sqrt(L * var_d + d^2 * var_L)
                cantidad a pedir       Q  = max(0, S - stock actual)
              d y var_d son diarias; var_d sale del error de pronostico de la validacion.
Fase 6 (OE4)  Simulacion en los origenes de evaluacion (no usados para elegir): quiebres,
              tasa de cumplimiento e inventario medio de la politica propuesta frente al
              metodo actual (pedir lo vendido el mes anterior, segun la gerencia), con y sin
              stock de seguridad, y frente al promedio de 6 meses con stock de seguridad.

Entradas
    data/processed/panel_mensual.parquet, segmentos.parquet, cv_predicciones.parquet
    src/parametros_inventario.py
Salidas
    data/processed/plan_reposicion.parquet
    outputs/plan_reposicion.csv                    plan por producto-sucursal (para la empresa)
    outputs/reports/04_pronostico_inventario.md    resumen (solo agregados)

Uso
    python src/04_pronostico_inventario.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from parametros_inventario import (
    DIAS_POR_MES,
    DIAS_REVISION,
    LEAD_TIME_DIAS,
    NIVEL_SERVICIO,
    lead_time_medio_y_desvio,
)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUTS_DIR = ROOT / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"

SERIE = ["centro", "sku"]
MODELO_SIN_CLASE = "Media"  # series con una sola venta: no se puede calcular CV^2
# Metodo actual de la empresa: se pide lo vendido el mes anterior (confirmado por la gerencia).
REFERENCIA = "Naive"
# Referencia adicional: promedio de los ultimos 6 meses (supuesto inicial del proyecto).
REFERENCIA_6M = "MetodoActual6M"

# Validacion de las series vigentes con historia corta (< 24 meses).
CANDIDATOS_CORTA = ["IMAPA", "ADIDA", "Media", "MetodoActual6M", "SES", "Naive", "TSB"]
N_VENTANAS_CORTA = 4
MESES_MINIMOS_VALIDACION_CORTA = N_VENTANAS_CORTA + 3  # al menos 3 meses de entrenamiento
Z =stats.norm.ppf(NIVEL_SERVICIO)


def cargar(nombre: str, alias: str):
    ruta = Path(__file__).resolve().parent / nombre
    spec = importlib.util.spec_from_file_location(alias, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


f3b = cargar("03b_estrategias_seleccion.py", "fase3b")
f3 = f3b.f3


# --------------------------------------------------------------------------------------
# Fase 4 - Pronostico
# --------------------------------------------------------------------------------------
def series_vigentes(panel: pd.DataFrame, seg: pd.DataFrame) -> pd.DataFrame:
    """Formato (unique_id, ds, y) de todas las series vigentes, con el mes faltante interpolado."""
    claves = seg.loc[seg["estado_ciclo"] == "vigente", SERIE]
    df = panel.merge(claves, on=SERIE)
    df = df[df["activo"]].sort_values(SERIE + ["periodo"]).copy()
    df["demanda"] = (
        df.groupby(SERIE)["demanda"].transform(lambda s: s.interpolate(limit_direction="both")).fillna(0)
    )
    df["unique_id"] = df["centro"] + "|" + df["sku"]
    return df.rename(columns={"periodo": "ds", "demanda": "y"})[["unique_id", "ds", "y"]]


def mapa_clases(metricas: pd.DataFrame, clases: pd.Series, individuales: list[str]) -> pd.Series:
    """Modelo de mejor rango promedio en cada clase de demanda (estrategia 'modelo por clase')."""
    por_clase = f3b.asignar(metricas, clases, individuales)["modelo por clase"]
    return por_clase.groupby(clases.reindex(por_clase.index)).first()


def pronosticar(series: pd.DataFrame, alias: list[str]) -> pd.DataFrame:
    from statsforecast import StatsForecast
    from statsforecast.models import Naive

    catalogo = {getattr(m, "alias", str(m)): m for m in f3.modelos()}
    sf = StatsForecast(
        models=[catalogo[a] for a in alias], freq="MS", n_jobs=-1, fallback_model=Naive()
    )
    fc = sf.forecast(df=series, h=1)
    if "unique_id" not in fc.columns:
        fc = fc.reset_index()
    return fc.set_index("unique_id")


def validar_cortas(series: pd.DataFrame) -> pd.DataFrame:
    """Validacion de origen movil (h = 1) de las series vigentes con historia corta."""
    from statsforecast import StatsForecast
    from statsforecast.models import Naive

    catalogo = {getattr(m, "alias", str(m)): m for m in f3.modelos()}
    sf = StatsForecast(
        models=[catalogo[a] for a in CANDIDATOS_CORTA], freq="MS", n_jobs=-1, fallback_model=Naive()
    )
    cv = sf.cross_validation(df=series, h=1, step_size=1, n_windows=N_VENTANAS_CORTA)
    if "unique_id" not in cv.columns:
        cv = cv.reset_index()
    return cv


def error_mensual(metricas: pd.DataFrame, asignacion: pd.Series) -> pd.Series:
    """Desviacion del error de pronostico a 1 mes (RMSE) del modelo asignado a cada serie."""
    mse = metricas.pivot(index="unique_id", columns="modelo", values="mse")
    return np.sqrt(f3b.tomar(mse, asignacion))


# --------------------------------------------------------------------------------------
# Fase 5 - Politica (R, S)
# --------------------------------------------------------------------------------------
def politica(demanda_mes: pd.Series, sigma_mes: pd.Series, centro: pd.Series) -> pd.DataFrame:
    lt = {c: lead_time_medio_y_desvio(c) for c in LEAD_TIME_DIAS}
    L = centro.map({c: v[0] for c, v in lt.items()})
    sL = centro.map({c: v[1] for c, v in lt.items()})
    d = demanda_mes.clip(lower=0) / DIAS_POR_MES
    var_d = sigma_mes.fillna(0) ** 2 / DIAS_POR_MES  # errores diarios independientes
    P = DIAS_REVISION + L
    ss = Z * np.sqrt(P * var_d + d ** 2 * sL ** 2)
    return pd.DataFrame(
        {
            "lead_time_dias": L,
            "periodo_proteccion_dias": P,
            "demanda_diaria": d,
            "stock_seguridad": ss,
            "nivel_reposicion_S": d * P + ss,
            "punto_reorden": d * L + Z * np.sqrt(L * var_d + d ** 2 * sL ** 2),
            "nivel_sin_ss": d * P,
        },
        index=demanda_mes.index,
    )


# --------------------------------------------------------------------------------------
# Fase 6 - Simulacion
# --------------------------------------------------------------------------------------
def valor_por_fila(cv: pd.DataFrame, modelo_por_serie: pd.Series) -> pd.Series:
    cols = cv.columns.get_indexer(cv["unique_id"].map(modelo_por_serie))
    vals = cv.to_numpy()[np.arange(len(cv)), np.clip(cols, 0, None)]
    return pd.Series(np.where(cols >= 0, vals, np.nan), index=cv.index).astype(float)


def simular(cv_ev: pd.DataFrame, configuraciones: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cada fila es un ciclo de reposicion: el nivel S se fija con el pronostico del mes y se
    enfrenta a la demanda real de ese mes durante el periodo de proteccion (R + L)."""
    centro = cv_ev["unique_id"].str.split("|").str[0]
    resumen, por_serie = {}, {}
    for nombre, (modelo, sigma, con_ss) in configuraciones.items():
        pred = valor_por_fila(cv_ev, modelo)
        pol = politica(pred, cv_ev["unique_id"].map(sigma), centro)
        nivel = pol["nivel_reposicion_S"] if con_ss else pol["nivel_sin_ss"]
        d_real = cv_ev["y"] / DIAS_POR_MES
        demanda_P = d_real * pol["periodo_proteccion_dias"]
        faltante = (demanda_P - nivel).clip(lower=0)
        inventario = (nivel - d_real * (pol["lead_time_dias"] + DIAS_REVISION / 2)).clip(lower=0)
        con_demanda = demanda_P > 0
        resumen[nombre] = {
            "ciclos": len(cv_ev),
            "% ciclos con quiebre": 100 * (faltante[con_demanda] > 0).mean(),
            "tasa de cumplimiento (%)": 100 * (1 - faltante.sum() / demanda_P.sum()),
            "unidades faltantes": faltante.sum(),
            "inventario medio (unidades)": inventario.sum() / cv_ev["cutoff"].nunique(),
            "dias de cobertura": inventario.sum() / d_real.sum() if d_real.sum() else np.nan,
        }
        por_serie[nombre] = pd.DataFrame({"faltante": faltante, "inventario": inventario}).groupby(
            cv_ev["unique_id"]
        ).sum()
    return pd.DataFrame(resumen).T, por_serie


def configuraciones_simulacion(propuesto: pd.Series, m_sel: pd.DataFrame) -> dict:
    """Politicas que se comparan en la simulacion (las tres primeras, en este orden, las usa la Figura)."""
    actual = pd.Series(REFERENCIA, index=propuesto.index)
    prom6 = pd.Series(REFERENCIA_6M, index=propuesto.index)
    return {
        "propuesto: modelo por clase + SS": (propuesto, error_mensual(m_sel, propuesto), True),
        "metodo actual + SS": (actual, error_mensual(m_sel, actual), True),
        "metodo actual sin SS": (actual, error_mensual(m_sel, actual), False),
        "promedio 6 meses + SS": (prom6, error_mensual(m_sel, prom6), True),
    }


def wilcoxon(a: pd.Series, b: pd.Series) -> str:
    dif = (a - b).dropna()
    if (dif != 0).sum() == 0:
        return "sin diferencias"
    return f"{stats.wilcoxon(dif, alternative='less').pvalue:.2e}"


# --------------------------------------------------------------------------------------
def bloque(titulo: str, contenido) -> str:
    if isinstance(contenido, (pd.DataFrame, pd.Series)):
        contenido = contenido.to_string(float_format=lambda v: f"{v:,.2f}")
    return f"## {titulo}\n\n```\n{contenido}\n```\n"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(PROCESSED_DIR / "panel_mensual.parquet")
    seg = pd.read_parquet(PROCESSED_DIR / "segmentos.parquet")
    seg["unique_id"] = seg["centro"] + "|" + seg["sku"]
    cv = pd.read_parquet(PROCESSED_DIR / "cv_predicciones.parquet")
    cv = f3b.solo_horizonte(cv, 1).reset_index(drop=True)
    individuales = [c for c in cv.columns if c not in f3b.ID_COLS]

    clases = seg.set_index("unique_id")["clase_demanda"]
    torneo = f3.preparar_series(panel, seg)
    cortes = np.sort(cv["cutoff"].unique())
    sel, ev = cortes[: f3.N_VENTANAS_SELECCION], cortes[f3.N_VENTANAS_SELECCION:]
    m_todas = f3.calcular_metricas(cv, torneo)
    m_sel = f3.calcular_metricas(cv[cv["cutoff"].isin(sel)], torneo)

    # ---------------- Fase 4 ----------------
    mapa = mapa_clases(m_todas, clases, individuales)
    vig = seg[seg["estado_ciclo"] == "vigente"].set_index("unique_id")
    vig["segmento"] = np.where(vig["modelable"], "torneo (>= 24 meses)", "historia corta")
    series = series_vigentes(panel, seg)

    # Historia corta: validacion propia (los productos de catalogo rotativo decaen tras su
    # lanzamiento, y la regla del torneo no se puede trasladar sin evidencia).
    cortas = vig.index[~vig["modelable"]]
    largo = series.groupby("unique_id").size()
    validables = largo.index[(largo >= MESES_MINIMOS_VALIDACION_CORTA) & largo.index.isin(cortas)]
    serie_c = series[series["unique_id"].isin(validables)]
    cv_c = validar_cortas(serie_c)
    m_c = f3.calcular_metricas(cv_c, serie_c)
    mapa_corta = mapa_clases(m_c, clases, CANDIDATOS_CORTA)
    rango_c = m_c.pivot(index="unique_id", columns="modelo", values="rmsse")[CANDIDATOS_CORTA]
    mejor_corta = rango_c.rank(axis=1).mean().idxmin()

    # En historia corta se usa un solo modelo, el mejor de su validacion: la eleccion por
    # clase se apoya en muy pocas series validadas y se aplicaria sobre todo a lanzamientos
    # de menos de 7 meses, que no se pueden validar.
    vig["modelo"] = np.where(
        vig["modelable"], vig["clase_demanda"].map(mapa).fillna(MODELO_SIN_CLASE), mejor_corta
    )
    validadas_por_clase = clases.reindex(cv_c["unique_id"].unique()).value_counts(dropna=False)
    no_disponibles = set(vig["modelo"]) - {getattr(m, "alias", str(m)) for m in f3.modelos()}
    vig.loc[vig["modelo"].isin(no_disponibles), "modelo"] = MODELO_SIN_CLASE

    fc = pronosticar(series, sorted(set(vig["modelo"]) | {REFERENCIA}))
    mes_objetivo = fc["ds"].max()
    vig["pronostico"] = f3b.tomar(fc.drop(columns="ds"), vig["modelo"]).clip(lower=0)
    vig["pronostico_metodo_actual"] = fc[REFERENCIA].reindex(vig.index).clip(lower=0)

    # Error de pronostico: RMSE de la validacion de cada serie (torneo o historia corta);
    # las series demasiado cortas para validar usan el error relativo mediano de su clase.
    sigma = pd.concat([error_mensual(m_todas, vig["modelo"]), error_mensual(m_c, vig["modelo"])])
    sigma = sigma[~sigma.index.duplicated()].dropna()
    media_y = pd.concat([cv.groupby("unique_id")["y"].mean(), cv_c.groupby("unique_id")["y"].mean()])
    media_y = media_y[~media_y.index.duplicated()]
    relativo = (sigma / media_y.reindex(sigma.index).replace(0, np.nan)).replace(np.inf, np.nan)
    rel_clase = relativo.groupby(clases.reindex(relativo.index)).median()
    sigma_rel = vig["pronostico"] * vig["clase_demanda"].map(rel_clase).fillna(relativo.median())
    vig["sigma_mensual"] = sigma.reindex(vig.index).fillna(sigma_rel)
    vig["origen_sigma"] = np.where(
        vig.index.isin(sigma.index), "validacion propia", "error relativo de su clase"
    )

    resumen_corta = pd.DataFrame(
        {
            "rmsse_mediana": rango_c.median(),
            "rango_promedio": rango_c.rank(axis=1).mean(),
        }
    ).sort_values("rango_promedio")

    # ---------------- Fase 5 ----------------
    plan = vig.join(politica(vig["pronostico"], vig["sigma_mensual"], vig["centro"]))
    fin = panel["periodo"].max()
    stock = panel[panel["periodo"] == fin].assign(unique_id=lambda d: d["centro"] + "|" + d["sku"])
    plan["stock_actual"] = stock.set_index("unique_id")["stock_disponible"].reindex(plan.index)
    plan["stock_conocido"] = plan["stock_actual"].notna()
    disponible = plan["stock_actual"].fillna(0).clip(lower=0)
    for col in ["stock_seguridad", "nivel_reposicion_S", "punto_reorden"]:
        plan[col] = np.ceil(plan[col])
    plan["cantidad_pedir"] = np.ceil((plan["nivel_reposicion_S"] - disponible).clip(lower=0))
    plan["alerta"] = np.select(
        [
            ~plan["stock_conocido"],
            disponible <= plan["punto_reorden"],
            disponible < plan["nivel_reposicion_S"],
        ],
        [
            "stock desconocido (verificar inventario)",
            "urgente (bajo el punto de reorden)",
            "reponer en el proximo surtido",
        ],
        "sin pedido",
    )
    columnas = ["centro", "sku", "segmento", "clase_demanda", "abc", "modelo", "pronostico",
                "pronostico_metodo_actual", "sigma_mensual", "origen_sigma", "lead_time_dias",
                "periodo_proteccion_dias", "stock_seguridad", "punto_reorden", "nivel_reposicion_S",
                "stock_actual", "cantidad_pedir", "alerta"]
    plan = plan[columnas].reset_index()
    plan.to_parquet(PROCESSED_DIR / "plan_reposicion.parquet", index=False)
    plan.to_csv(OUTPUTS_DIR / "plan_reposicion.csv", index=False, encoding="utf-8-sig")

    # ---------------- Fase 6 ----------------
    mapa_sel = mapa_clases(m_sel, clases, individuales)
    propuesto = clases.reindex(cv["unique_id"].unique()).map(mapa_sel).fillna(MODELO_SIN_CLASE)
    cv_ev = cv[cv["cutoff"].isin(ev)].reset_index(drop=True)
    configuraciones = configuraciones_simulacion(propuesto, m_sel)
    sim, por_serie = simular(cv_ev, configuraciones)
    nombres = list(configuraciones)

    def prueba(medida: str, otra: int) -> str:
        return wilcoxon(por_serie[nombres[0]][medida], por_serie[nombres[otra]][medida])

    pruebas = pd.Series(
        {
            "faltantes: propuesto < actual + SS": prueba("faltante", 1),
            "faltantes: propuesto < actual sin SS": prueba("faltante", 2),
            "faltantes: propuesto < promedio 6 meses + SS": prueba("faltante", 3),
            "inventario: propuesto < actual + SS": prueba("inventario", 1),
            "inventario: propuesto < promedio 6 meses + SS": prueba("inventario", 3),
        },
        name="p Wilcoxon (unilateral, pareada por serie)",
    )

    # ---------------- Reporte ----------------
    parametros = pd.Series(
        {
            "mes pronosticado": f"{mes_objetivo:%Y-%m}",
            "periodo de revision R (dias)": DIAS_REVISION,
            "nivel de servicio": f"{NIVEL_SERVICIO:.0%} (z = {Z:.3f})",
            "origenes de evaluacion (simulacion)": f"{pd.Timestamp(ev.min()):%Y-%m} a {pd.Timestamp(ev.max()):%Y-%m}",
            "series inactivas sin reposicion": int((seg["estado_ciclo"] != "vigente").sum()),
        }
    )
    lead = pd.DataFrame(
        {c: {"L medio (dias)": lead_time_medio_y_desvio(c)[0], "desvio L (dias)": lead_time_medio_y_desvio(c)[1]}
         for c in LEAD_TIME_DIAS}
    ).T
    por_centro = plan.groupby("centro").agg(
        series=("sku", "size"),
        pronostico_total=("pronostico", "sum"),
        pronostico_metodo_actual=("pronostico_metodo_actual", "sum"),
        stock_seguridad=("stock_seguridad", "sum"),
        nivel_S=("nivel_reposicion_S", "sum"),
        stock_actual=("stock_actual", "sum"),
        cantidad_pedir=("cantidad_pedir", "sum"),
        stock_desconocido=("stock_actual", lambda s: int(s.isna().sum())),
    )
    secciones = [
        "# Fases 4 a 6 - Pronostico, inventario y validacion\n",
        bloque("Parametros", parametros),
        bloque("Lead time por centro (distribucion uniforme)", lead),
        bloque("Modelo asignado por clase de demanda (torneo, todas las ventanas)", mapa.rename("modelo")),
        bloque(
            f"Validacion de las series con historia corta ({cv_c['unique_id'].nunique()} series con "
            f">= {MESES_MINIMOS_VALIDACION_CORTA} meses, {N_VENTANAS_CORTA} origenes moviles)",
            resumen_corta,
        ),
        bloque(
            f"Historia corta: mejor modelo por clase (solo informativo; se usa {mejor_corta} para todas)",
            pd.DataFrame({"mejor modelo": mapa_corta, "series validadas": validadas_por_clase}),
        ),
        bloque("Series vigentes por segmento y modelo", pd.crosstab(plan["modelo"], plan["segmento"], margins=True)),
        bloque("Resumen del plan por centro (unidades)", por_centro),
        bloque(
            "Pronostico total por segmento y modelo: propuesto frente al metodo actual (unidades)",
            plan.groupby(["segmento", "modelo"])[["pronostico", "pronostico_metodo_actual"]]
            .sum()
            .assign(diferencia_pct=lambda t: 100 * (t["pronostico"] / t["pronostico_metodo_actual"] - 1)),
        ),
        bloque("Alertas por centro", pd.crosstab(plan["alerta"], plan["centro"], margins=True)),
        bloque("Alertas por clase ABC", pd.crosstab(plan["alerta"], plan["abc"], margins=True)),
        bloque(
            f"Simulacion de la politica (series del torneo, {len(ev)} meses no usados para elegir)",
            sim,
        ),
        bloque("Pruebas de Wilcoxon por serie", pruebas),
    ]
    reporte = "\n".join(secciones)
    (REPORTS_DIR / "04_pronostico_inventario.md").write_text(reporte, encoding="utf-8")
    print(reporte)
    print(f"Plan guardado en {OUTPUTS_DIR / 'plan_reposicion.csv'}")


if __name__ == "__main__":
    main()
