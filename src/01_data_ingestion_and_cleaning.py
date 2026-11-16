"""Fase 1 - Ingesta y limpieza de datos.

Lee el Excel de ventas e inventario (una hoja por mes), valida su estructura, limpia los
registros y construye el panel mensual centro x SKU que usan las fases siguientes.

Entradas
    data/raw/*.xlsx (o data/*.xlsx si aun no se movio el archivo)

Salidas
    data/processed/ventas_long.parquet    registros limpios, una fila por linea del reporte
    data/processed/panel_mensual.parquet  serie mensual completa por centro x SKU
    data/processed/perfil_series.parquet  ADI, CV2 y clase de demanda por serie
    data/processed/catalogo_sku.csv       descripcion vigente por SKU
    outputs/reports/01_calidad_datos.md   resumen de calidad (solo agregados, sin filas)

Uso
    python src/01_data_ingestion_and_cleaning.py
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Configuracion
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_DIRS = [ROOT / "data" / "raw", ROOT / "data"]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "outputs" / "reports"

PERIODO_INICIO = "2024-01"
PERIODO_FIN = "2026-06"

# Valores de la columna "Tipo" que cuentan como demanda. El resto se reporta pero no se suma.
TIPOS_DEMANDA = ["VENTA"]

# Centros que no son sedes reales (regex sobre el nombre del centro).
PATRON_CENTROS_EXCLUIDOS = r"PRUEBA"

# Meses sin venta al cierre a partir de los cuales una serie se considera inactiva.
MESES_INACTIVIDAD = 6

# Historia minima (meses activos) para que una serie vigente entre al torneo de modelos.
MESES_MINIMOS_MODELADO = 24

# Umbrales de Syntetos-Boylan para clasificar la demanda.
ADI_UMBRAL = 1.32
CV2_UMBRAL = 0.49

MESES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}

# Encabezado normalizado del Excel -> nombre canonico.
COLUMNAS = {
    "centro": "centro_id",
    "nombre centro": "centro",
    "tipo": "tipo",
    "codigo": "sku",
    "descripcion": "descripcion",
    "cantidad fac/dev": "cantidad",
    "cantidad disponible": "stock_disponible",
    "cantidad apartada": "stock_apartado",
}
NUMERICAS = ["cantidad", "stock_disponible", "stock_apartado"]
SERIE = ["centro", "sku"]
CLAVE = ["periodo", "centro", "sku"]


# --------------------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------------------
def norm(valor) -> str:
    """Minusculas, sin acentos y con espacios colapsados; '' para nulos."""
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return ""
    texto = unicodedata.normalize("NFKD", str(valor))
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", texto).strip().lower()


def sku_a_texto(valor):
    """Codigo de producto como texto estable (1532.0 -> '1532')."""
    if pd.isna(valor):
        return pd.NA
    if isinstance(valor, (int, np.integer)):
        return str(int(valor))
    if isinstance(valor, (float, np.floating)) and float(valor).is_integer():
        return str(int(valor))
    return str(valor).strip()


def periodo_de_hoja(nombre: str):
    """'enero 2024' -> Timestamp('2024-01-01'); None si la hoja no es mensual."""
    m = re.fullmatch(r"([a-z]+)\s+(\d{4})", norm(nombre))
    if not m or m.group(1) not in MESES:
        return None
    return pd.Timestamp(year=int(m.group(2)), month=MESES[m.group(1)], day=1)


def bloque(titulo: str, contenido) -> str:
    if isinstance(contenido, (pd.DataFrame, pd.Series)):
        contenido = contenido.to_string()
    return f"## {titulo}\n\n```\n{contenido}\n```\n"


def localizar_excel() -> Path:
    for carpeta in RAW_DIRS:
        archivos = sorted(p for p in carpeta.glob("*.xls[xm]") if not p.name.startswith("~$"))
        if len(archivos) == 1:
            return archivos[0]
        if len(archivos) > 1:
            raise SystemExit(f"Hay varios Excel en {carpeta}; deja solo el de ventas.")
    raise SystemExit("No se encontro ningun Excel en data/raw ni en data/.")


# --------------------------------------------------------------------------------------
# Ingesta
# --------------------------------------------------------------------------------------
def parsear_hoja(raw: pd.DataFrame, periodo: pd.Timestamp, nombre: str):
    """Ubica la fila de encabezados y devuelve las columnas canonicas de la hoja."""
    primera_col = raw.iloc[:15, 0].map(norm)
    filas_encabezado = primera_col[primera_col == "centro"].index
    if len(filas_encabezado) == 0:
        raise ValueError(f"Hoja '{nombre}': no se encontro la fila de encabezados.")
    h = filas_encabezado[0]

    encabezados = [norm(x) for x in raw.iloc[h]]
    datos = raw.iloc[h + 1:].copy()
    datos.columns = range(datos.shape[1])

    conocidas: dict[int, str] = {}
    for i, e in enumerate(encabezados):
        if e in COLUMNAS and COLUMNAS[e] not in conocidas.values():
            conocidas[i] = COLUMNAS[e]
    faltantes = sorted(set(COLUMNAS.values()) - set(conocidas.values()))
    if faltantes:
        raise ValueError(f"Hoja '{nombre}': faltan columnas {faltantes}.")

    celdas_extra = int(sum(datos[i].notna().sum() for i in datos.columns if i not in conocidas))
    datos = datos[list(conocidas)].rename(columns=conocidas).dropna(how="all")
    datos["periodo"] = periodo
    datos["hoja"] = nombre
    return datos, celdas_extra


def leer_excel(path: Path):
    hojas = pd.read_excel(path, sheet_name=None, header=None, dtype=object, engine="openpyxl")
    partes, omitidas, extras = [], [], {}
    for nombre, raw in hojas.items():
        periodo = periodo_de_hoja(nombre)
        if periodo is None:
            omitidas.append(f"{nombre} ({raw.shape[0]} filas x {raw.shape[1]} columnas)")
            continue
        datos, celdas_extra = parsear_hoja(raw, periodo, nombre)
        partes.append(datos)
        if celdas_extra:
            extras[nombre] = celdas_extra
    if not partes:
        raise SystemExit("Ninguna hoja tiene nombre de mes ('enero 2024', ...).")
    return pd.concat(partes, ignore_index=True), omitidas, extras


# --------------------------------------------------------------------------------------
# Limpieza
# --------------------------------------------------------------------------------------
def limpiar(df: pd.DataFrame):
    """Estandariza tipos y texto. Devuelve el dataframe limpio y las metricas de calidad."""
    calidad = {"filas_crudas": len(df), "nulos_crudos": df[list(COLUMNAS.values())].isna().sum()}

    for col in ["centro", "tipo", "descripcion"]:
        df[col] = (
            df[col].astype("string").str.replace(r"\s+", " ", regex=True).str.strip().str.upper()
        )
    df["sku"] = df["sku"].map(sku_a_texto).astype("string")
    df["centro_id"] = pd.to_numeric(df["centro_id"], errors="coerce").astype("Int64")

    no_numericos = {}
    for col in NUMERICAS:
        convertido = pd.to_numeric(df[col], errors="coerce")
        no_numericos[col] = int((convertido.isna() & df[col].notna()).sum())
        df[col] = convertido.astype("float64")
    calidad["no_numericos"] = pd.Series(no_numericos)

    # Filas sin centro o sin codigo: totales, notas al pie, etc.
    sin_clave = df["sku"].isna() | df["centro"].isna()
    calidad["filas_sin_clave"] = int(sin_clave.sum())
    df = df[~sin_clave].copy()

    # Duplicados de clave: si son copias exactas o lineas con cantidades distintas cambia el tratamiento.
    calidad["duplicados"] = int(df.duplicated(["periodo", "centro", "tipo", "sku"], keep=False).sum())
    calidad["duplicados_identicos"] = int(
        df.duplicated(["periodo", "centro", "tipo", "sku", "descripcion"] + NUMERICAS, keep=False).sum()
    )
    return df, calidad


# --------------------------------------------------------------------------------------
# Panel mensual
# --------------------------------------------------------------------------------------
def construir_panel(df: pd.DataFrame, periodos: pd.DatetimeIndex):
    """Serie mensual completa por centro x SKU.

    - demanda: suma de `cantidad` de los tipos en TIPOS_DEMANDA; negativos netos -> 0.
    - Un mes sin registro posterior a la primera aparicion del SKU en el centro es demanda 0.
    - Los meses anteriores a la primera aparicion quedan en NaN (el producto no existia).
    - Un mes en que el centro no tiene ninguna fila es un faltante del reporte, no demanda 0:
      queda en NaN con mes_faltante = True.
    - stock: maximo por mes (el reporte repite el mismo saldo en cada linea del SKU).
    """
    ventas = df[df["tipo"].isin(TIPOS_DEMANDA)]
    demanda = ventas.groupby(CLAVE)["cantidad"].sum(min_count=1).rename("demanda")
    stock = df.groupby(CLAVE)[["stock_disponible", "stock_apartado"]].max()
    obs = stock.join(demanda, how="outer").reset_index()
    obs["observado"] = True

    pares = obs[SERIE].drop_duplicates()
    malla = pares.merge(pd.DataFrame({"periodo": periodos}), how="cross")
    panel = malla.merge(obs, on=CLAVE, how="left")
    panel["observado"] = panel["observado"].eq(True)

    presentes = df[["centro", "periodo"]].drop_duplicates().assign(centro_con_datos=True)
    panel = panel.merge(presentes, on=["centro", "periodo"], how="left")
    panel["mes_faltante"] = panel.pop("centro_con_datos").isna()

    fechas_obs = panel["periodo"].where(panel["observado"]).groupby([panel["centro"], panel["sku"]])
    panel["primer_periodo"] = fechas_obs.transform("min")
    panel["ultimo_periodo"] = fechas_obs.transform("max")
    panel["activo"] = panel["periodo"] >= panel["primer_periodo"]

    negativos = int((panel["demanda"] < 0).sum())
    valido = panel["activo"] & ~panel["mes_faltante"]
    panel["demanda"] = panel["demanda"].fillna(0).clip(lower=0).where(valido)
    panel["sin_registro"] = valido & ~panel["observado"]
    panel["posible_quiebre"] = (
        panel["activo"] & panel["demanda"].eq(0) & panel["stock_disponible"].le(0)
    )
    panel = panel.sort_values(SERIE + ["periodo"]).reset_index(drop=True)
    return panel, negativos


def perfilar_series(panel: pd.DataFrame, fin: pd.Timestamp) -> pd.DataFrame:
    """ADI, CV2 y clase de demanda (Syntetos-Boylan) sobre los meses activos de cada serie,
    mas el estado de ciclo de vida segun los meses transcurridos desde la ultima venta."""
    activo = panel[panel["activo"]]
    perfil = activo.groupby(SERIE)["demanda"].agg(
        n_meses="count",  # no cuenta los meses faltantes (NaN)
        n_meses_con_venta=lambda s: int((s > 0).sum()),
        demanda_total="sum",
    )
    ultimo = activo.groupby(SERIE)["ultimo_periodo"].first()
    perfil["ultimo_periodo"] = ultimo
    perfil["meses_sin_venta_al_cierre"] = (fin.year - ultimo.dt.year) * 12 + (fin.month - ultimo.dt.month)
    perfil["estado_ciclo"] = np.where(
        perfil["meses_sin_venta_al_cierre"] < MESES_INACTIVIDAD, "vigente", "inactivo"
    )
    positivos = activo[activo["demanda"] > 0].groupby(SERIE)["demanda"].agg(
        media_positiva="mean",
        std_positiva=lambda s: float(s.std(ddof=0)),
    )
    perfil = perfil.join(positivos)
    perfil["adi"] = perfil["n_meses"] / perfil["n_meses_con_venta"].replace(0, np.nan)
    perfil["cv2"] = (perfil["std_positiva"] / perfil["media_positiva"]) ** 2

    adi_bajo = perfil["adi"] < ADI_UMBRAL
    cv2_bajo = perfil["cv2"] < CV2_UMBRAL
    perfil["clase_demanda"] = np.select(
        [
            perfil["n_meses_con_venta"].eq(0),
            adi_bajo & cv2_bajo,
            ~adi_bajo & cv2_bajo,
            adi_bajo & ~cv2_bajo,
        ],
        ["sin_venta", "suave", "intermitente", "erratica"],
        default="irregular",
    )
    return perfil.reset_index()


# --------------------------------------------------------------------------------------
# Reporte
# --------------------------------------------------------------------------------------
def resumen_centros(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["centro_id", "centro"], dropna=False)
        .agg(filas=("sku", "size"), skus=("sku", "nunique"), meses=("periodo", "nunique"))
        .reset_index()
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    path = localizar_excel()
    periodos = pd.date_range(PERIODO_INICIO, PERIODO_FIN, freq="MS")

    crudo, omitidas, extras = leer_excel(path)
    df, calidad = limpiar(crudo)

    excluido = df["centro"].str.contains(PATRON_CENTROS_EXCLUIDOS, regex=True, na=False)
    centros_excluidos = resumen_centros(df[excluido])
    df = df[~excluido].copy()

    fuera_de_rango = ~df["periodo"].isin(periodos)
    periodos_presentes = set(df.loc[~fuera_de_rango, "periodo"])
    periodos_faltantes = [p.strftime("%Y-%m") for p in periodos if p not in periodos_presentes]
    df = df[~fuera_de_rango].copy()

    panel, negativos = construir_panel(df, periodos)
    perfil = perfilar_series(panel, periodos[-1])
    catalogo = (
        df.sort_values("periodo")
        .groupby("sku")["descripcion"]
        .agg(descripcion="last", n_descripciones="nunique")
        .reset_index()
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_DIR / "ventas_long.parquet", index=False)
    panel.to_parquet(PROCESSED_DIR / "panel_mensual.parquet", index=False)
    perfil.to_parquet(PROCESSED_DIR / "perfil_series.parquet", index=False)
    catalogo.to_csv(PROCESSED_DIR / "catalogo_sku.csv", index=False, encoding="utf-8-sig")

    # ---- Reporte de calidad: solo agregados -------------------------------------------
    activo = panel[panel["activo"] & ~panel["mes_faltante"]]
    meses_faltantes = (
        panel.loc[panel["mes_faltante"], ["centro", "periodo"]]
        .drop_duplicates()
        .assign(periodo=lambda d: d["periodo"].dt.strftime("%Y-%m"))
        .groupby("centro")["periodo"]
        .agg(", ".join)
    )
    modelable = (perfil["estado_ciclo"] == "vigente") & (perfil["n_meses"] >= MESES_MINIMOS_MODELADO)
    por_hoja = df.groupby("periodo").agg(
        filas=("sku", "size"), skus=("sku", "nunique"), centros=("centro", "nunique")
    )
    por_hoja.index = por_hoja.index.strftime("%Y-%m")
    tipos = df.groupby("tipo", dropna=False)["cantidad"].agg(
        filas="size",
        suma="sum",
        negativos=lambda s: int((s < 0).sum()),
        ceros=lambda s: int((s == 0).sum()),
    )
    stock_distintos = df.groupby(SERIE)["stock_disponible"].agg(["nunique", "size"])
    con_historia = stock_distintos[stock_distintos["size"] >= 6]
    pct_stock_constante = 100 * (con_historia["nunique"] <= 1).mean() if len(con_historia) else np.nan
    largo = pd.cut(
        perfil["n_meses"], bins=[0, 5, 11, 23, 29, 30], labels=["1-5", "6-11", "12-23", "24-29", "30"]
    )

    generales = pd.Series(
        {
            "archivo": path.name,
            "hojas mensuales leidas": crudo["hoja"].nunique(),
            "hojas omitidas": "; ".join(omitidas) or "ninguna",
            "meses esperados sin datos": ", ".join(periodos_faltantes) or "ninguno",
            "filas crudas": calidad["filas_crudas"],
            "filas sin centro o sin codigo (descartadas)": calidad["filas_sin_clave"],
            "filas fuera del rango de fechas (descartadas)": int(fuera_de_rango.sum()),
            "filas duplicadas en periodo+centro+tipo+sku": calidad["duplicados"],
            "  de ellas, copias exactas en todas las columnas": calidad["duplicados_identicos"],
            "hojas con celdas fuera de las 8 columnas": len(extras),
            "celdas fuera de las 8 columnas (ignoradas)": sum(extras.values()),
            "filas limpias": len(df),
            "SKUs distintos": df["sku"].nunique(),
            "SKUs con mas de una descripcion": int((catalogo["n_descripciones"] > 1).sum()),
            "series centro x SKU": len(perfil),
        }
    )
    panel_stats = pd.Series(
        {
            "filas del panel (series x 30 meses)": len(panel),
            "meses-serie activos": len(activo),
            "% meses activos con demanda 0": round(100 * activo["demanda"].eq(0).mean(), 1),
            "% meses activos sin registro en el Excel": round(100 * activo["sin_registro"].mean(), 1),
            "meses-serie con demanda neta negativa (llevados a 0)": negativos,
            "meses-serie con posible quiebre (demanda 0 y stock <= 0)": int(panel["posible_quiebre"].sum()),
            "% series con stock disponible identico todos los meses": round(pct_stock_constante, 1),
            f"series vigentes (venta en los ultimos {MESES_INACTIVIDAD} meses)": int((perfil["estado_ciclo"] == "vigente").sum()),
            f"series modelables (vigentes y >= {MESES_MINIMOS_MODELADO} meses)": int(modelable.sum()),
        }
    )

    secciones = [
        "# Fase 1 - Calidad de datos\n",
        bloque("Resumen general", generales),
        bloque("Nulos por columna (antes de limpiar)", calidad["nulos_crudos"]),
        bloque("Valores no numericos en columnas de cantidad", calidad["no_numericos"]),
        bloque("Centros incluidos", resumen_centros(df)),
        bloque("Centros excluidos", centros_excluidos if len(centros_excluidos) else "ninguno"),
        bloque("Valores de la columna Tipo", tipos),
        bloque("Filas, SKUs y centros por mes", por_hoja),
        bloque("Meses sin ninguna fila en un centro (tratados como faltantes)", meses_faltantes if len(meses_faltantes) else "ninguno"),
        bloque("Panel mensual", panel_stats),
        bloque("Series por estado de ciclo de vida y clase de demanda", pd.crosstab(perfil["estado_ciclo"], perfil["clase_demanda"], margins=True)),
        bloque("Series modelables por centro y clase de demanda", pd.crosstab(perfil.loc[modelable, "centro"], perfil.loc[modelable, "clase_demanda"], margins=True)),
        bloque("Series por centro y clase de demanda", pd.crosstab(perfil["centro"], perfil["clase_demanda"], margins=True)),
        bloque("Series por centro y meses de historia", pd.crosstab(perfil["centro"], largo, margins=True)),
    ]
    reporte = "\n".join(secciones)
    (REPORTS_DIR / "01_calidad_datos.md").write_text(reporte, encoding="utf-8")
    print(reporte)
    print(f"Reporte guardado en {REPORTS_DIR / '01_calidad_datos.md'}")


if __name__ == "__main__":
    main()
