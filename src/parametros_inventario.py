"""Parametros de la politica de inventario de Empresa Alfa S.A. (entrada de la Fase 5).

Fuente: gerencia de la empresa, comunicacion del 10 y 11 de noviembre de 2026.

Flujo de abastecimiento:
    Mexico -> deposito central (Santa Cruz) -> centros de La Paz, Cochabamba y Santa Cruz.
    Marketing arma el catalogo segun el stock del deposito y el deposito distribuye a cada
    centro segun su movimiento de ventas. El sistema modela la reposicion deposito -> centro.
"""

# Tiempo de reposicion deposito -> centro, en dias (minimo, maximo).
LEAD_TIME_DIAS = {
    "LA PAZ": (2.0, 3.0),
    "COCHABAMBA": (1.0, 2.0),
    "SANTA CRUZ": (0.5, 1.0),
}

# Revision periodica: la empresa surte dos veces al mes, en dias fijos y escalonados
# (lunes La Paz, martes Cochabamba, miercoles Santa Cruz).
DIAS_REVISION = 15
DIA_SURTIDO = {"LA PAZ": "lunes", "COCHABAMBA": "martes", "SANTA CRUZ": "miercoles"}

# Nivel de servicio objetivo (probabilidad de no tener quiebre en un ciclo).
NIVEL_SERVICIO = 0.95

# El catalogo cambia cada 2 meses e incorpora productos nuevos.
MESES_CICLO_CATALOGO = 2

DIAS_POR_MES = 30.4


def lead_time_medio_y_desvio(centro: str) -> tuple[float, float]:
    """Media y desviacion del lead time asumiendo distribucion uniforme entre minimo y maximo."""
    minimo, maximo = LEAD_TIME_DIAS[centro]
    return (minimo + maximo) / 2, (maximo - minimo) / 12 ** 0.5
