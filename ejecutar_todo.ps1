# Ejecuta el sistema completo, en orden, con el entorno virtual del proyecto.
# Uso (desde la carpeta del proyecto):
#     powershell -ExecutionPolicy Bypass -File ejecutar_todo.ps1
# Para saltar el torneo (la fase mas lenta, ~1 h 15 min) si ya existe su resultado:
#     powershell -ExecutionPolicy Bypass -File ejecutar_todo.ps1 -SinTorneo

param([switch]$SinTorneo)

$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw 'No se encontro .venv. Crea el entorno segun el README.' }

$fases = @(
    '01_data_ingestion_and_cleaning.py',
    '02_eda_segmentacion.py',
    '03_torneo_modelos.py',
    '03b_estrategias_seleccion.py',
    '04_pronostico_inventario.py',
    '05_figuras.py'
)

foreach ($script in $fases) {
    if ($SinTorneo -and $script -eq '03_torneo_modelos.py') {
        Write-Host "Omitido: $script (se usa el resultado anterior)"
        continue
    }
    Write-Host ''
    Write-Host "=== $script ==="
    $inicio = Get-Date
    & $python (Join-Path $PSScriptRoot "src\$script")
    if ($LASTEXITCODE -ne 0) { throw "Fallo $script (codigo $LASTEXITCODE)" }
    Write-Host ("Terminado en {0:N0} s" -f ((Get-Date) - $inicio).TotalSeconds)
}
Write-Host ''
Write-Host 'Listo: reportes en outputs\reports, figuras en outputs\figures y plan en outputs\plan_reposicion.csv'
