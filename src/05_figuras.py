"""Fase 7 - Figuras para el capitulo de resultados.

Genera, a partir de los archivos procesados, las figuras de las Fases 3 a 6:

    05_ranking_modelos.png      rango promedio de RMSSE con la diferencia critica de Nemenyi
    05_estrategias.png          evaluacion honesta de las estrategias de seleccion
    05_simulacion.png           quiebres e inventario de las politicas comparadas (OE4)
    05_ejemplo_serie.png        una serie A con sus pronosticos (producto anonimizado)
    05_alertas.png              alertas del plan de reposicion por centro

Entradas
    data/processed/panel_mensual.parquet, segmentos.parquet, cv_predicciones.parquet,
    plan_reposicion.parquet (Fase 4)
Salidas
    outputs/figures/05_*.png

Uso
    python src/05_figuras.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "outputs" / "figures"

# Paleta: gris para el contexto y un solo acento para lo que el titulo destaca.
ACENTO = "#2a78d6"
REFERENCIA_COLOR = "#0b0b0b"
GRIS = "#b9b8b2"
GRIS_OSCURO = "#52514e"
TEXTO = "#0b0b0b"
ESTADOS = {  # paleta de estado, siempre con etiqueta
    "urgente (bajo el punto de reorden)": "#d03b3b",
    "reponer en el proximo surtido": "#fab219",
    "sin pedido": "#0ca30c",
    "stock desconocido (verificar inventario)": "#d6d5cf",
}
ETIQUETA_ESTADO = {
    "urgente (bajo el punto de reorden)": "Urgente",
    "reponer en el proximo surtido": "Reponer en el próximo surtido",
    "sin pedido": "Sin pedido",
    "stock desconocido (verificar inventario)": "Stock desconocido",
}
NOMBRES = {
    "MetodoActual6M": "Promedio de 6 meses",
    "Naive": "Método actual (mes anterior)",
    "Media": "Media histórica",
    "SeasonalNaive": "Estacional ingenuo anual",
    "Bimestral": "Estacional ingenuo bimestral",
}
ESTRATEGIAS = {
    "modelo por clase": "Modelo por clase de demanda",
    "combinacion": "Combinación de líderes",
    "metodo actual": "Método actual (mes anterior)",
    "promedio 6 meses": "Promedio de 6 meses",
    "hibrida por clase": "Híbrida por clase",
    "hibrida por combinacion": "Híbrida por combinación",
    "ganador por serie": "Ganador individual por serie",
}

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.edgecolor": GRIS_OSCURO,
        "axes.labelcolor": GRIS_OSCURO,
        "xtick.color": GRIS_OSCURO,
        "ytick.color": GRIS_OSCURO,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def cargar(nombre: str, alias: str):
    ruta = Path(__file__).resolve().parent / nombre
    spec = importlib.util.spec_from_file_location(alias, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


f4 = cargar("04_pronostico_inventario.py", "fase4")
f3b, f3 = f4.f3b, f4.f3


def coma(v: float, dec: int = 2) -> str:
    return f"{v:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def guardar(fig, nombre: str) -> Path:
    ruta = FIG_DIR / nombre
    fig.savefig(ruta)
    plt.close(fig)
    return ruta


# --------------------------------------------------------------------------------------
def fig_ranking(m_todas: pd.DataFrame) -> Path:
    rangos, _, p, cd, n = f3.prueba_friedman(m_todas)
    rangos = rangos.sort_values()
    mejor, ref = rangos.iloc[0], rangos[f4.REFERENCIA]
    y = np.arange(len(rangos))[::-1]

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.axvspan(mejor, mejor + cd, color=GRIS, alpha=0.25, lw=0)
    ax.text(mejor + cd / 2, -0.55, f"Diferencia crítica\nde Nemenyi = {coma(cd)}",
            fontsize=8, color=GRIS_OSCURO, ha="center", va="bottom")
    for yi, (modelo, r) in zip(y, rangos.items()):
        es_mejor, es_ref = r == mejor, modelo == f4.REFERENCIA
        color = ACENTO if es_mejor else REFERENCIA_COLOR if es_ref else GRIS
        ax.plot(r, yi, "o", ms=8, color=color, zorder=3)
        etiqueta = coma(r)
        if es_ref:
            etiqueta += f"  (a {coma(ref - mejor)} del primero)"
        ax.text(r + 0.2, yi, etiqueta, va="center", fontsize=8.5,
                color=TEXTO if es_ref or es_mejor else GRIS_OSCURO)
    ax.set_yticks(y)
    ax.set_yticklabels([NOMBRES.get(m, m) for m in rangos.index])
    for t, m in zip(ax.get_yticklabels(), rangos.index):
        if m == f4.REFERENCIA or rangos[m] == mejor:
            t.set_fontweight("bold")
            t.set_color(TEXTO)
    ax.set_xlim(np.floor(mejor) - 0.3, np.ceil(rangos.max()) + 1.2)
    ax.set_ylim(-1.2, len(rangos) - 0.4)
    ax.grid(axis="x", color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Rango promedio del RMSSE (1 = mejor modelo en la serie)")
    ax.set_title(f"{NOMBRES.get(rangos.index[0], rangos.index[0])} encabeza el ranking; "
                 f"el método actual queda a {coma(ref - mejor)} puntos", loc="left", pad=22)
    ax.text(0, 1.02, f"Rango promedio de RMSSE en {n} series · Friedman p = {p:.1e} · "
            "franja: sin diferencia significativa con el primero",
            transform=ax.transAxes, fontsize=8.5, color=GRIS_OSCURO)
    return guardar(fig, "05_ranking_modelos.png")


def fig_estrategias(evaluacion: pd.DataFrame) -> Path:
    t = evaluacion.copy()
    t["rmsse_mediana"] = t["rmsse_mediana"].astype(float)
    t = t.sort_values("rmsse_mediana", ascending=False)
    base = t.loc["metodo actual", "rmsse_mediana"]
    y = np.arange(len(t))

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for yi, (nombre, fila) in zip(y, t.iterrows()):
        p = fila["p Wilcoxon (< metodo actual)"]
        significativa = p != "" and float(p) < 0.05
        color = ACENTO if significativa else REFERENCIA_COLOR if nombre == "metodo actual" else GRIS
        ax.barh(yi, fila["rmsse_mediana"], height=0.55, color=color)
        nota = coma(fila["rmsse_mediana"], 3)
        if nombre != "metodo actual":
            nota += (f"   mejora en {coma(float(fila['% series mejor que metodo actual']), 1)} % de las series"
                     f" · p = {float(p):.1e}" if p != "" else "")
        ax.text(max(fila["rmsse_mediana"], base) + 0.03, yi, nota, va="center", fontsize=8.5,
                color=TEXTO if significativa else GRIS_OSCURO)
    ax.axvline(base, color=REFERENCIA_COLOR, lw=1, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([ESTRATEGIAS.get(n, n) for n in t.index])
    ax.set_xlim(0, t["rmsse_mediana"].max() * 2.3)
    ax.set_xlabel("RMSSE mediano en los meses de evaluación (menor es mejor)")
    p = t["p Wilcoxon (< metodo actual)"]
    ganadoras = [ESTRATEGIAS.get(n, n).lower() for n in t.index if p[n] != "" and float(p[n]) < 0.05]
    if len(ganadoras) == 1:
        titulo = f"Solo la estrategia «{ganadoras[0]}» mejora al método actual con significancia"
    elif ganadoras:
        titulo = f"{len(ganadoras)} estrategias mejoran al método actual con significancia"
    else:
        titulo = "Ninguna estrategia mejora al método actual con significancia"
    ax.set_title(titulo, loc="left", pad=22)
    ax.text(0, 1.03, "Azul: Wilcoxon unilateral p < 0,05 frente al método actual · "
            "línea discontinua: método actual", transform=ax.transAxes, fontsize=8.5, color=GRIS_OSCURO)
    return guardar(fig, "05_estrategias.png")


def fig_simulacion(sim: pd.DataFrame, n_series: int, periodo: str) -> Path:
    nombres = {
        "propuesto: modelo por clase + SS": "Propuesta\n(por clase + SS)",
        "metodo actual + SS": "Método actual\n+ SS",
        "metodo actual sin SS": "Método actual\nsin SS",
        "promedio 6 meses + SS": "Promedio\n6 meses + SS",
    }
    colores = [ACENTO, GRIS, GRIS_OSCURO, GRIS][: len(sim)]
    x = np.arange(len(sim))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    paneles = [
        (a1, "% ciclos con quiebre", "Ciclos con quiebre de stock (%)", 1),
        (a2, "inventario medio (unidades)", "Inventario medio (unidades)", 0),
    ]
    for ax, col, titulo, dec in paneles:
        valores = sim[col].astype(float)
        ax.bar(x, valores, width=0.6, color=colores)
        for xi, v in zip(x, valores):
            ax.text(xi, v, coma(v, dec), ha="center", va="bottom", fontsize=9, color=TEXTO)
        ax.set_xticks(x)
        ax.set_xticklabels([nombres.get(n, n) for n in sim.index], fontsize=8)
        ax.set_title(titulo, fontsize=10.5)
        ax.set_ylim(0, valores.max() * 1.18)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    q = sim["% ciclos con quiebre"].astype(float)
    inv = sim["inventario medio (unidades)"].astype(float)
    cierre = ("el modelo por clase mantiene ese servicio con menos inventario" if inv.iloc[0] < inv.iloc[1]
              else "el modelo por clase requiere más inventario")
    fig.suptitle(
        f"El stock de seguridad baja los quiebres del {coma(q.iloc[2], 1)} % al {coma(q.iloc[0], 1)} %; {cierre}",
        x=0.02, ha="left", fontsize=11.5, fontweight="bold",
    )
    fig.text(0.02, -0.04, f"Simulación en los meses de evaluación ({periodo}), {n_series} series del torneo. "
             "Objetivo de servicio: 95 % (5 % de ciclos con quiebre).", fontsize=8, color=GRIS_OSCURO)
    fig.tight_layout()
    return guardar(fig, "05_simulacion.png")


def fig_ejemplo(series: pd.DataFrame, cv: pd.DataFrame, seg: pd.DataFrame, plan: pd.DataFrame,
                mapa_sel: pd.Series) -> Path:
    candidatas = seg[seg["modelable"] & (seg["abc"] == "A")]
    fila = candidatas.sort_values("volumen_reciente", ascending=False).iloc[0]
    uid = fila["unique_id"]
    modelo = mapa_sel.get(fila["clase_demanda"], f4.MODELO_SIN_CLASE)
    hist = series[series["unique_id"] == uid]
    val = cv[cv["unique_id"] == uid].sort_values("ds")
    futuro = plan[plan["unique_id"] == uid].iloc[0]
    mes = val["ds"].max() + pd.offsets.MonthBegin(1)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(hist["ds"], hist["y"], color=GRIS_OSCURO, lw=1.6, marker="o", ms=3.5, label="Venta real")
    ax.plot(val["ds"], val[modelo].clip(lower=0), color=ACENTO, lw=2, marker="o", ms=5,
            label=f"Propuesta ({NOMBRES.get(modelo, modelo)}), validación")
    ax.plot(val["ds"], val[f4.REFERENCIA].clip(lower=0), color=REFERENCIA_COLOR, lw=1.2, ls="--",
            label="Método actual, validación")
    ax.plot([mes], [futuro["pronostico"]], marker="D", ms=8, color=ACENTO, ls="none")
    ax.annotate(f"Pronóstico {mes:%m/%Y}: {coma(futuro['pronostico'], 0)}",
                (mes, futuro["pronostico"]), xytext=(8, 10), textcoords="offset points",
                fontsize=8.5, color=TEXTO)
    ax.axvspan(val["ds"].min() - pd.Timedelta(days=15), mes + pd.Timedelta(days=15),
               color=GRIS, alpha=0.12, lw=0)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    ax.set_ylabel("Unidades por mes")
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=3, bbox_to_anchor=(0, -0.12))
    ax.set_title(f"Ejemplo: producto A de mayor volumen en {fila['centro'].title()} "
                 f"(clase {fila['clase_demanda']})\nZona sombreada: orígenes de validación y mes pronosticado",
                 loc="left")
    return guardar(fig, "05_ejemplo_serie.png")


def fig_alertas(plan: pd.DataFrame) -> Path:
    tabla = pd.crosstab(plan["centro"], plan["alerta"]).reindex(columns=list(ESTADOS), fill_value=0)
    tabla = tabla.loc[tabla.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(8, 3))
    izquierda = np.zeros(len(tabla))
    for estado, color in ESTADOS.items():
        v = tabla[estado].to_numpy()
        ax.barh(tabla.index.str.title(), v, left=izquierda, color=color, height=0.55,
                edgecolor="white", linewidth=2, label=ETIQUETA_ESTADO[estado])
        for i, (l, n) in enumerate(zip(izquierda, v)):
            if n > 0:
                ax.text(l + n / 2, i, str(n), ha="center", va="center", fontsize=8 if n < 10 else 8.5,
                        color=TEXTO)
        izquierda += v
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Series producto-sucursal vigentes")
    ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.25))
    urgentes = int((plan["alerta"] == list(ESTADOS)[0]).sum())
    reponer = int((plan["alerta"] == list(ESTADOS)[1]).sum())
    ax.set_title(f"{urgentes} series requieren reposición urgente y {reponer} en el próximo surtido",
                 loc="left")
    return guardar(fig, "05_alertas.png")


# --------------------------------------------------------------------------------------
def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(PROCESSED_DIR / "panel_mensual.parquet")
    seg = pd.read_parquet(PROCESSED_DIR / "segmentos.parquet")
    seg["unique_id"] = seg["centro"] + "|" + seg["sku"]
    plan = pd.read_parquet(PROCESSED_DIR / "plan_reposicion.parquet")
    cv_base = f3b.solo_horizonte(pd.read_parquet(PROCESSED_DIR / "cv_predicciones.parquet"), 1)
    cv_base = cv_base.reset_index(drop=True)
    individuales = [c for c in cv_base.columns if c not in f3b.ID_COLS]
    cv = f3b.agregar_combinaciones(cv_base)

    clases = seg.set_index("unique_id")["clase_demanda"]
    series = f3.preparar_series(panel, seg)
    cortes = np.sort(cv["cutoff"].unique())
    sel, ev = cortes[: f3.N_VENTANAS_SELECCION], cortes[f3.N_VENTANAS_SELECCION:]
    m_todas = f3.calcular_metricas(cv_base, series)
    m_sel_c = f3.calcular_metricas(cv[cv["cutoff"].isin(sel)], series)
    m_ev_c = f3.calcular_metricas(cv[cv["cutoff"].isin(ev)], series)
    m_sel = m_sel_c[m_sel_c["modelo"].isin(individuales)]

    evaluacion = f3b.evaluar(f3b.asignar(m_sel_c, clases, individuales), m_ev_c)

    mapa_sel = f4.mapa_clases(m_sel, clases, individuales)
    propuesto = clases.reindex(cv_base["unique_id"].unique()).map(mapa_sel).fillna(f4.MODELO_SIN_CLASE)
    cv_ev = cv_base[cv_base["cutoff"].isin(ev)].reset_index(drop=True)
    sim, _ = f4.simular(cv_ev, f4.configuraciones_simulacion(propuesto, m_sel))

    rutas = [
        fig_ranking(m_todas),
        fig_estrategias(evaluacion),
        fig_simulacion(
            sim,
            cv_ev["unique_id"].nunique(),
            f"{pd.Timestamp(ev.min()):%m/%Y} a {pd.Timestamp(ev.max()):%m/%Y}",
        ),
        fig_ejemplo(series, cv_base, seg, plan, mapa_sel),
        fig_alertas(plan),
    ]
    for ruta in rutas:
        print(f"Figura guardada: {ruta}")


if __name__ == "__main__":
    main()
