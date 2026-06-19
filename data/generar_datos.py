"""
Generador de datos de ejemplo para CBB Predictor.
------------------------------------------------
Este script crea un archivo CSV ('cbb_games.csv') con juegos simulados
de baloncesto universitario (College Basketball) que parecen realistas.

NO necesitas ejecutar este archivo para usar el proyecto: el CSV ya viene
incluido. Solo se incluye por si quieres regenerar o cambiar los datos.

Para ejecutarlo:
    python data/generar_datos.py
"""

import csv
import random
from datetime import date, timedelta

# Hacemos los datos "reproducibles": siempre saldrán los mismos números.
random.seed(42)

# ------------------------------------------------------------------
# 1) Definimos los equipos y una "fuerza" base para cada uno.
#    Una fuerza más alta = mejor equipo (gana más y anota más).
# ------------------------------------------------------------------
EQUIPOS = {
    "Duke":          88,
    "Kansas":        87,
    "Gonzaga":       86,
    "Kentucky":      85,
    "North Carolina":84,
    "UCLA":          83,
    "Houston":       86,
    "Purdue":        84,
    "Baylor":        82,
    "Arizona":       83,
    "Tennessee":     81,
    "Michigan St":   80,
    "Villanova":     79,
    "Texas":         80,
    "Auburn":        81,
    "Marquette":     78,
}

NOMBRES = list(EQUIPOS.keys())

# ------------------------------------------------------------------
# 2) Simulamos una temporada con varios partidos por equipo.
# ------------------------------------------------------------------
NUM_JUEGOS = 250          # cantidad de partidos a generar
fecha_inicio = date(2024, 11, 4)   # típico arranque de temporada CBB

def simular_puntos(fuerza_local, fuerza_visitante):
    """
    Devuelve (puntos_local, puntos_visitante) de forma realista.
    - Sumamos una ventaja de cancha (home advantage) de ~3.5 puntos al local.
    - La diferencia de fuerza influye en el marcador.
    - Agregamos aleatoriedad para que no sea predecible al 100%.
    """
    ventaja_local = 3.5

    # Puntos base ~ media de 72 con variación segun la fuerza del equipo.
    base_local = fuerza_local + ventaja_local + random.gauss(0, 7)
    base_visit = fuerza_visitante + random.gauss(0, 7)

    # Escalamos a un rango típico de puntos de baloncesto universitario (60-90).
    puntos_local = int(round(base_local - 12 + random.gauss(0, 4)))
    puntos_visit = int(round(base_visit - 12 + random.gauss(0, 4)))

    # Evitamos empates (en baloncesto siempre hay un ganador).
    if puntos_local == puntos_visit:
        puntos_local += random.choice([-1, 1])

    # Mantenemos los puntos en un rango razonable.
    puntos_local = max(48, min(105, puntos_local))
    puntos_visit = max(48, min(105, puntos_visit))
    return puntos_local, puntos_visit


filas = []
fecha_actual = fecha_inicio

for i in range(NUM_JUEGOS):
    # Elegimos dos equipos distintos al azar.
    local, visitante = random.sample(NOMBRES, 2)

    pl, pv = simular_puntos(EQUIPOS[local], EQUIPOS[visitante])

    filas.append({
        "fecha": fecha_actual.isoformat(),
        "equipo_local": local,
        "equipo_visitante": visitante,
        "puntos_local": pl,
        "puntos_visitante": pv,
        # 1 = ganó el local, 0 = ganó el visitante. Esto es lo que predecimos.
        "gano_local": 1 if pl > pv else 0,
    })

    # Avanzamos la fecha cada 1-2 días para simular un calendario.
    fecha_actual += timedelta(days=random.choice([1, 1, 2]))

# ------------------------------------------------------------------
# 3) Guardamos el CSV.
# ------------------------------------------------------------------
RUTA_CSV = "data/cbb_games.csv"
with open(RUTA_CSV, "w", newline="", encoding="utf-8") as f:
    columnas = ["fecha", "equipo_local", "equipo_visitante",
                "puntos_local", "puntos_visitante", "gano_local"]
    escritor = csv.DictWriter(f, fieldnames=columnas)
    escritor.writeheader()
    escritor.writerows(filas)

print(f"Listo: {len(filas)} juegos guardados en {RUTA_CSV}")
