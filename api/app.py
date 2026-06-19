"""
app.py - API REST con Flask
===========================

Una API muy sencilla para pedir predicciones desde cualquier programa,
página web o herramienta como Postman / curl.

Cómo ejecutarla (desde la carpeta principal del proyecto):

    python api/app.py

Luego abre en tu navegador:  http://localhost:5000

Endpoints (rutas) disponibles:
    GET  /              -> mensaje de bienvenida e instrucciones
    GET  /equipos       -> lista de equipos disponibles
    GET  /predecir?local=Duke&visitante=Kansas   -> predicción rápida
    POST /predecir      -> predicción enviando JSON
"""

import os
import sys

# Permite importar 'model.py' que está en la carpeta src/.
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(RAIZ, "src"))

from flask import Flask, request, jsonify  # noqa: E402
import model  # noqa: E402

app = Flask(__name__)

# Cargamos el modelo UNA sola vez al iniciar la API (es más rápido).
RUTA_MODELO = os.path.join(RAIZ, "src", "modelo_entrenado.pkl")
try:
    PAQUETE = model.cargar_modelo(RUTA_MODELO)
except FileNotFoundError:
    PAQUETE = None


@app.route("/")
def inicio():
    """Página de bienvenida con instrucciones de uso."""
    return jsonify({
        "mensaje": "Bienvenido a la API de CBB Predictor",
        "como_usar": {
            "ver_equipos": "GET /equipos",
            "predecir_rapido": "GET /predecir?local=Duke&visitante=Kansas",
            "predecir_json": "POST /predecir con {'local': 'Duke', 'visitante': 'Kansas'}",
        },
        "modelo_cargado": PAQUETE is not None,
    })


@app.route("/equipos")
def equipos():
    """Devuelve la lista de equipos que el modelo conoce."""
    if PAQUETE is None:
        return jsonify({"error": "Modelo no entrenado. Ejecuta: python src/train.py"}), 503
    return jsonify({"equipos": model.lista_equipos(PAQUETE)})


@app.route("/predecir", methods=["GET", "POST"])
def predecir():
    """
    Predice el ganador de un partido.
    Acepta los datos por GET (parámetros en la URL) o por POST (JSON).
    """
    if PAQUETE is None:
        return jsonify({"error": "Modelo no entrenado. Ejecuta: python src/train.py"}), 503

    # Obtenemos 'local' y 'visitante' venga como venga (GET o POST).
    if request.method == "POST":
        datos = request.get_json(silent=True) or {}
        local = datos.get("local")
        visitante = datos.get("visitante")
    else:
        local = request.args.get("local")
        visitante = request.args.get("visitante")

    if not local or not visitante:
        return jsonify({
            "error": "Faltan parámetros. Indica 'local' y 'visitante'.",
            "ejemplo": "/predecir?local=Duke&visitante=Kansas",
        }), 400

    try:
        resultado = model.predecir(PAQUETE, local, visitante)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify(resultado)


if __name__ == "__main__":
    # host="0.0.0.0" permite acceder desde otros dispositivos en la red.
    app.run(host="0.0.0.0", port=5000, debug=True)
