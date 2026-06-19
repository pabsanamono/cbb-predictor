"""
model.py - El "cerebro" del CBB Predictor
==========================================

Aquí está toda la lógica para:
  1) Leer los datos de los juegos (CSV).
  2) Calcular estadísticas útiles de cada equipo (promedios recientes,
     ventaja de jugar en casa, etc.). A esto le llamamos "features"
     (características que el modelo usa para aprender).
  3) Entrenar un modelo de Machine Learning sencillo.
  4) Predecir quién ganaría un partido entre dos equipos.

Usamos pandas (para manejar tablas de datos) y scikit-learn
(la librería de Machine Learning más popular y fácil de Python).

Mejoras incluidas respecto a un modelo básico:
  - Distingue equipo LOCAL y VISITANTE (home/away).
  - Usa PROMEDIOS MÓVILES (rolling averages): el rendimiento reciente
    de cada equipo en sus últimos N partidos, no toda la temporada.
  - Calcula racha de victorias y diferencia de puntos promedio.
"""

import os
import joblib  # para guardar/cargar el modelo entrenado en disco
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Cuántos partidos recientes usamos para los promedios móviles.
VENTANA_RECIENTE = 5

# Nombres de las columnas (features) que el modelo usará para aprender.
FEATURES = [
    "dif_pts_anotados",     # dif. de puntos anotados promedio (local - visitante)
    "dif_pts_recibidos",    # dif. de puntos recibidos promedio (local - visitante)
    "dif_racha",            # dif. de racha de victorias reciente
    "dif_win_rate",         # dif. de % de victorias en la temporada
]


# ======================================================================
# 1) CARGAR Y PREPARAR LOS DATOS
# ======================================================================
def cargar_datos(ruta_csv):
    """Lee el CSV y lo ordena por fecha (importante para promedios móviles)."""
    df = pd.read_csv(ruta_csv, parse_dates=["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    return df


def _historial_largo(df):
    """
    Convierte la tabla de partidos en una tabla "larga" donde cada fila
    es la actuación de UN equipo en UN partido. Así es más fácil calcular
    promedios móviles por equipo.
    """
    locales = pd.DataFrame({
        "fecha": df["fecha"],
        "equipo": df["equipo_local"],
        "pts_anotados": df["puntos_local"],
        "pts_recibidos": df["puntos_visitante"],
        "gano": (df["puntos_local"] > df["puntos_visitante"]).astype(int),
    })
    visitantes = pd.DataFrame({
        "fecha": df["fecha"],
        "equipo": df["equipo_visitante"],
        "pts_anotados": df["puntos_visitante"],
        "pts_recibidos": df["puntos_local"],
        "gano": (df["puntos_visitante"] > df["puntos_local"]).astype(int),
    })
    largo = pd.concat([locales, visitantes]).sort_values("fecha")
    return largo


def _stats_por_equipo(largo):
    """
    Para cada equipo calcula sus estadísticas ANTES de cada partido,
    usando solo partidos pasados (rolling). Esto evita "hacer trampa"
    mirando el resultado del partido que queremos predecir.
    """
    largo = largo.sort_values("fecha").copy()

    def calc(grupo):
        grupo = grupo.sort_values("fecha")
        # .shift(1) = usar solo datos ANTERIORES al partido actual.
        grupo["pa_prom"] = (
            grupo["pts_anotados"].shift(1)
            .rolling(VENTANA_RECIENTE, min_periods=1).mean()
        )
        grupo["pr_prom"] = (
            grupo["pts_recibidos"].shift(1)
            .rolling(VENTANA_RECIENTE, min_periods=1).mean()
        )
        grupo["racha"] = (
            grupo["gano"].shift(1)
            .rolling(VENTANA_RECIENTE, min_periods=1).mean()
        )
        grupo["win_rate"] = grupo["gano"].shift(1).expanding().mean()
        return grupo

    return largo.groupby("equipo", group_keys=False).apply(calc)


def construir_features(df):
    """
    A partir de la tabla de partidos, construye la tabla final con las
    'features' (X) y la respuesta correcta (y = gano_local).
    """
    largo = _historial_largo(df)
    stats = _stats_por_equipo(largo)

    # Separamos de nuevo en estadísticas del local y del visitante.
    # Usamos (fecha, equipo) como clave para volver a unir.
    stats_idx = stats.set_index(["fecha", "equipo"])

    filas = []
    for _, juego in df.iterrows():
        clave_local = (juego["fecha"], juego["equipo_local"])
        clave_visit = (juego["fecha"], juego["equipo_visitante"])
        try:
            sl = stats_idx.loc[clave_local]
            sv = stats_idx.loc[clave_visit]
        except KeyError:
            continue
        # Si un equipo aparece dos veces el mismo día, tomamos la primera fila.
        if isinstance(sl, pd.DataFrame):
            sl = sl.iloc[0]
        if isinstance(sv, pd.DataFrame):
            sv = sv.iloc[0]

        filas.append({
            "dif_pts_anotados":  (sl["pa_prom"] or 0) - (sv["pa_prom"] or 0),
            "dif_pts_recibidos": (sl["pr_prom"] or 0) - (sv["pr_prom"] or 0),
            "dif_racha":         (sl["racha"] or 0) - (sv["racha"] or 0),
            "dif_win_rate":      (sl["win_rate"] or 0) - (sv["win_rate"] or 0),
            "gano_local":        int(juego["gano_local"]),
        })

    tabla = pd.DataFrame(filas).fillna(0)
    return tabla


# ======================================================================
# 2) ENTRENAR EL MODELO
# ======================================================================
def entrenar(df):
    """
    Entrena el modelo con los datos y devuelve (modelo, precision).
    Usamos una Regresión Logística: simple, rápida y fácil de entender.
    """
    tabla = construir_features(df)
    X = tabla[FEATURES]
    y = tabla["gano_local"]

    # Pipeline = escalar los datos + modelo, todo en un solo objeto.
    modelo = Pipeline([
        ("escalador", StandardScaler()),
        ("clasificador", LogisticRegression(max_iter=1000)),
    ])

    # Dividimos en entrenamiento (80%) y prueba (20%) para medir qué tan bien
    # predice partidos que NO vio durante el entrenamiento.
    n = len(X)
    corte = int(n * 0.8)
    X_train, X_test = X.iloc[:corte], X.iloc[corte:]
    y_train, y_test = y.iloc[:corte], y.iloc[corte:]

    modelo.fit(X_train, y_train)
    precision = modelo.score(X_test, y_test)

    # Volvemos a entrenar con TODOS los datos para el modelo final.
    modelo.fit(X, y)
    return modelo, precision


# ======================================================================
# 3) GUARDAR Y CARGAR EL MODELO
# ======================================================================
def guardar_modelo(modelo, df, ruta="src/modelo_entrenado.pkl"):
    """
    Guarda el modelo y también las últimas estadísticas de cada equipo,
    para poder hacer predicciones nuevas sin recalcular todo.
    """
    largo = _historial_largo(df)
    stats = _stats_por_equipo(largo)

    # Tomamos la fila MÁS RECIENTE de cada equipo (su estado actual).
    ultimas = (
        stats.sort_values("fecha")
        .groupby("equipo")
        .agg(
            pa_prom=("pts_anotados", lambda s: s.tail(VENTANA_RECIENTE).mean()),
            pr_prom=("pts_recibidos", lambda s: s.tail(VENTANA_RECIENTE).mean()),
            racha=("gano", lambda s: s.tail(VENTANA_RECIENTE).mean()),
            win_rate=("gano", "mean"),
        )
    )
    paquete = {"modelo": modelo, "stats_equipos": ultimas}
    joblib.dump(paquete, ruta)
    return ruta


def cargar_modelo(ruta="src/modelo_entrenado.pkl"):
    """Carga el modelo entrenado desde disco."""
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encontró el modelo en '{ruta}'. "
            "Primero ejecuta: python src/train.py"
        )
    return joblib.load(ruta)


# ======================================================================
# 4) HACER UNA PREDICCIÓN
# ======================================================================
def predecir(paquete, equipo_local, equipo_visitante):
    """
    Predice el resultado de un partido entre dos equipos.
    Devuelve un diccionario con el ganador y las probabilidades.
    """
    modelo = paquete["modelo"]
    stats = paquete["stats_equipos"]

    if equipo_local not in stats.index:
        raise ValueError(f"Equipo desconocido: '{equipo_local}'")
    if equipo_visitante not in stats.index:
        raise ValueError(f"Equipo desconocido: '{equipo_visitante}'")

    sl = stats.loc[equipo_local]
    sv = stats.loc[equipo_visitante]

    X = pd.DataFrame([{
        "dif_pts_anotados":  sl["pa_prom"] - sv["pa_prom"],
        "dif_pts_recibidos": sl["pr_prom"] - sv["pr_prom"],
        "dif_racha":         sl["racha"] - sv["racha"],
        "dif_win_rate":      sl["win_rate"] - sv["win_rate"],
    }])[FEATURES]

    prob_local = float(modelo.predict_proba(X)[0][1])
    prob_visit = 1 - prob_local
    ganador = equipo_local if prob_local >= 0.5 else equipo_visitante

    return {
        "equipo_local": equipo_local,
        "equipo_visitante": equipo_visitante,
        "prob_gana_local": round(prob_local, 3),
        "prob_gana_visitante": round(prob_visit, 3),
        "ganador_predicho": ganador,
        "confianza": round(max(prob_local, prob_visit), 3),
    }


def lista_equipos(paquete):
    """Devuelve la lista de equipos que el modelo conoce."""
    return sorted(paquete["stats_equipos"].index.tolist())
