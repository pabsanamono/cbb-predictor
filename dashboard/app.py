"""
app.py - Dashboard interactivo con Streamlit
============================================

Una página web sencilla donde puedes:
  - Elegir dos equipos y ver quién ganaría (con probabilidades).
  - Ver gráficas de los datos de la temporada.

Cómo ejecutarlo (desde la carpeta principal del proyecto):

    streamlit run dashboard/app.py

Se abrirá automáticamente en tu navegador (normalmente en
http://localhost:8501).
"""

import os
import sys

# Permite importar 'model.py' que está en la carpeta src/.
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(RAIZ, "src"))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402
import model  # noqa: E402

# ----------------------------------------------------------------------
# Configuración general de la página
# ----------------------------------------------------------------------
st.set_page_config(page_title="CBB Predictor", page_icon="🏀", layout="wide")
st.title("🏀 CBB Predictor - Baloncesto Universitario")
st.caption("Predice el ganador de un partido usando Machine Learning.")

RUTA_MODELO = os.path.join(RAIZ, "src", "modelo_entrenado.pkl")
RUTA_CSV = os.path.join(RAIZ, "data", "cbb_games.csv")


# Usamos cache para no recargar el modelo/datos en cada interacción.
@st.cache_resource
def cargar_paquete():
    return model.cargar_modelo(RUTA_MODELO)


@st.cache_data
def cargar_datos():
    return model.cargar_datos(RUTA_CSV)


# ----------------------------------------------------------------------
# Verificamos que el modelo exista
# ----------------------------------------------------------------------
if not os.path.exists(RUTA_MODELO):
    st.error("⚠️ El modelo aún no está entrenado.\n\n"
             "Ejecuta primero en la terminal:  `python src/train.py`")
    st.stop()

paquete = cargar_paquete()
df = cargar_datos()
equipos = model.lista_equipos(paquete)

# ======================================================================
# SECCIÓN 1: Predicción de un partido
# ======================================================================
st.header("1. Predecir un partido")

col1, col2 = st.columns(2)
with col1:
    local = st.selectbox("Equipo LOCAL (juega en casa)", equipos, index=0)
with col2:
    visitante = st.selectbox("Equipo VISITANTE", equipos, index=1)

if st.button("🔮 Predecir ganador", type="primary"):
    if local == visitante:
        st.warning("Elige dos equipos diferentes.")
    else:
        r = model.predecir(paquete, local, visitante)
        st.success(f"**Ganador predicho: {r['ganador_predicho']}** "
                   f"(confianza {r['confianza']*100:.1f}%)")

        # Barras con las probabilidades de cada equipo.
        probs = pd.DataFrame({
            "Equipo": [f"{local} (local)", f"{visitante} (visitante)"],
            "Probabilidad de ganar (%)": [
                r["prob_gana_local"] * 100,
                r["prob_gana_visitante"] * 100,
            ],
        })
        fig = px.bar(probs, x="Equipo", y="Probabilidad de ganar (%)",
                     color="Equipo", text_auto=".1f", range_y=[0, 100])
        st.plotly_chart(fig, use_container_width=True)

# ======================================================================
# SECCIÓN 2: Estadísticas de la temporada
# ======================================================================
st.header("2. Estadísticas de la temporada")

# Métricas rápidas en la parte superior.
total_juegos = len(df)
victorias_local = int((df["puntos_local"] > df["puntos_visitante"]).sum())
prom_pts = (df["puntos_local"] + df["puntos_visitante"]).mean() / 2

m1, m2, m3 = st.columns(3)
m1.metric("Juegos totales", total_juegos)
m2.metric("Victorias del local", f"{victorias_local/total_juegos*100:.0f}%")
m3.metric("Puntos promedio por equipo", f"{prom_pts:.1f}")

# Gráfica: puntos promedio anotados por cada equipo.
largo = pd.concat([
    df[["equipo_local", "puntos_local"]].rename(
        columns={"equipo_local": "equipo", "puntos_local": "puntos"}),
    df[["equipo_visitante", "puntos_visitante"]].rename(
        columns={"equipo_visitante": "equipo", "puntos_visitante": "puntos"}),
])
prom_equipo = (largo.groupby("equipo")["puntos"].mean()
               .sort_values(ascending=False).reset_index())
prom_equipo.columns = ["Equipo", "Puntos promedio"]

fig2 = px.bar(prom_equipo, x="Equipo", y="Puntos promedio",
              title="Puntos promedio anotados por equipo",
              color="Puntos promedio", color_continuous_scale="Blues")
st.plotly_chart(fig2, use_container_width=True)

# Tabla con los últimos partidos.
st.subheader("Últimos 15 partidos")
ultimos = df.sort_values("fecha", ascending=False).head(15).copy()
ultimos["resultado"] = ultimos.apply(
    lambda x: f"{x['puntos_local']} - {x['puntos_visitante']}", axis=1)
st.dataframe(
    ultimos[["fecha", "equipo_local", "equipo_visitante", "resultado"]],
    use_container_width=True, hide_index=True,
)
