# Sistema predictivo de demanda e inventarios – Empresa Alfa S.A.

Código de la monografía del Diplomado en Estadística Aplicada (UMSS): pronóstico de la demanda
mensual por producto y sucursal, y política de reposición de inventario desde el depósito central
de Santa Cruz a los centros de La Paz, Cochabamba y Santa Cruz.

## Qué hace

| Fase | Script | Objetivo | Resultado principal |
| --- | --- | --- | --- |
| 1 | `src/01_data_ingestion_and_cleaning.py` | OE1 | Limpieza, panel mensual, clasificación Syntetos-Boylan |
| 2 | `src/02_eda_segmentacion.py` | OE1 | Segmentos (torneo, historia corta, inactivas), ABC, estacionalidad, figuras |
| 3 | `src/03_torneo_modelos.py` | OE2 | Torneo de 14 modelos con validación de origen móvil (12 orígenes, h = 1) |
| 3b | `src/03b_estrategias_seleccion.py` | OE2 | Comparación de estrategias de selección (Wilcoxon frente al método actual) |
| 4–6 | `src/04_pronostico_inventario.py` | OE2–OE4 | Pronóstico, política (R, S), alertas, simulación de quiebres |
| 7 | `src/05_figuras.py` | – | Figuras del capítulo de resultados |

Los parámetros de inventario (lead time por centro, revisión quincenal, nivel de servicio del 95 %)
están en `src/parametros_inventario.py`.

## Instalación (Windows)

Requiere Python 3.11.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

No hace falta activar el entorno: todos los comandos usan `.venv\Scripts\python.exe`.

## Datos

Colocar el Excel de ventas en `data/raw/` (o en `data/`). Los datos son privados y no forman parte
del repositorio ni del CD (`.gitignore` excluye `data/` y `outputs/`).

## Ejecución

Todo en orden:

```powershell
powershell -ExecutionPolicy Bypass -File ejecutar_todo.ps1
```

| Fase | Duración aproximada |
| --- | --- |
| 1, 2, 3b, 7 | segundos a 1 minuto cada una |
| 3 (torneo, incluye Prophet) | ~1 h 15 min |
| 4–6 | 2 a 5 minutos |

Si el torneo ya se corrió y solo cambian las fases siguientes:

```powershell
powershell -ExecutionPolicy Bypass -File ejecutar_todo.ps1 -SinTorneo
```

También se puede correr cada fase por separado, por ejemplo
`.venv\Scripts\python.exe src\04_pronostico_inventario.py`.

## Salidas

| Carpeta o archivo | Contenido |
| --- | --- |
| `data/processed/` | Tablas intermedias en Parquet (panel, segmentos, validación, asignación, plan) |
| `outputs/reports/` | Un reporte Markdown por fase, solo con agregados |
| `outputs/figures/` | Figuras de las Fases 2 y 7 |
| `outputs/plan_reposicion.csv` | Plan por producto y sucursal: pronóstico, stock de seguridad, punto de reorden, nivel S, cantidad a pedir y alerta |

## Decisiones metodológicas

- **Métrica:** RMSSE (el MAPE no está definido con ventas cero; MAE y MASE premian pronosticar cero).
- **Selección:** modelo por clase de demanda en las series con al menos 24 meses (IMAPA para suaves e
  irregulares, ADIDA para erráticas, media histórica para intermitentes). La selección individual
  por serie sobreajusta y no mejora al método actual.
- **Método actual:** según la gerencia, la empresa pide de cada producto lo vendido el mes anterior
  (modelo Naive); es la referencia de las pruebas de Wilcoxon y de la simulación. El promedio de los
  últimos 6 meses se mantiene como segunda referencia.
- **Historia corta:** promedio de los últimos 6 meses, que gana en su propia validación.
- **Inventario:** revisión periódica (R, S) con R = 15 días; el stock de seguridad combina el error
  de pronóstico y la variabilidad del lead time (distribución uniforme por centro).
- **Validación:** los modelos se eligen con los 8 primeros orígenes y se evalúan en los 4 últimos.
