"""
train.py - Entrena el modelo y lo guarda en disco
=================================================

Ejecuta este archivo UNA VEZ (o cada vez que cambies los datos) para
crear el archivo 'modelo_entrenado.pkl' que usan la API y el dashboard.

Cómo ejecutarlo (desde la carpeta principal del proyecto):

    python src/train.py
"""

import os
import sys

# Permite importar 'model.py' aunque ejecutes el script desde otra carpeta.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import model  # noqa: E402  (nuestro archivo model.py)

RUTA_CSV = "data/cbb_games.csv"
RUTA_MODELO = "src/modelo_entrenado.pkl"


def main():
    print("=" * 55)
    print("  CBB PREDICTOR - Entrenamiento del modelo")
    print("=" * 55)

    # 1) Cargar datos
    print(f"\n1) Leyendo datos de: {RUTA_CSV}")
    df = model.cargar_datos(RUTA_CSV)
    print(f"   -> {len(df)} juegos cargados.")

    # 2) Entrenar
    print("\n2) Entrenando el modelo...")
    modelo, precision = model.entrenar(df)
    print(f"   -> Precisión en datos de prueba: {precision*100:.1f}%")

    # 3) Guardar
    print(f"\n3) Guardando el modelo en: {RUTA_MODELO}")
    model.guardar_modelo(modelo, df, RUTA_MODELO)
    print("   -> Modelo guardado correctamente.")

    # 4) Ejemplo de predicción
    print("\n4) Ejemplo de predicción:")
    paquete = model.cargar_modelo(RUTA_MODELO)
    equipos = model.lista_equipos(paquete)
    if len(equipos) >= 2:
        resultado = model.predecir(paquete, equipos[0], equipos[1])
        print(f"   {resultado['equipo_local']} (local) vs "
              f"{resultado['equipo_visitante']} (visitante)")
        print(f"   Ganador predicho: {resultado['ganador_predicho']} "
              f"(confianza {resultado['confianza']*100:.1f}%)")

    print("\n¡Listo! Ya puedes usar la API o el dashboard.\n")


if __name__ == "__main__":
    main()
