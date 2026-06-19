# 🏀 CBB Predictor — Predictor de Baloncesto Universitario

Proyecto sencillo en **Python** que predice quién ganará un partido de
baloncesto universitario (College Basketball) usando **Machine Learning**.

Está pensado para **principiantes**: el código está comentado en español,
paso a paso, y todo funciona desde el primer momento sin configuraciones
complicadas.

---

## 📋 ¿Qué hace este proyecto?

1. Usa datos de partidos (un archivo CSV con ~250 juegos de ejemplo).
2. Entrena un modelo que aprende qué equipos suelen ganar.
3. Te permite predecir el resultado de un nuevo partido de 3 formas:
   - Desde la **terminal** (al entrenar).
   - Desde una **API REST** (Flask).
   - Desde un **dashboard web** interactivo (Streamlit).

---

## 📁 Estructura del proyecto

```
cbb-predictor/
├── data/
│   ├── cbb_games.csv        ← Datos de ejemplo (250 partidos)
│   └── generar_datos.py     ← Script que creó esos datos (opcional)
├── src/
│   ├── model.py             ← El "cerebro": lógica del modelo
│   └── train.py             ← Entrena y guarda el modelo
├── api/
│   └── app.py               ← API REST con Flask
├── dashboard/
│   └── app.py               ← Dashboard web con Streamlit
├── requirements.txt         ← Lista de librerías necesarias
├── .gitignore               ← Archivos que Git debe ignorar
└── README.md                ← Este archivo
```

---

## 🚀 Guía rápida (paso a paso)

> 💡 Necesitas tener **Python 3.9 o superior** instalado.
> Para comprobarlo, abre una terminal y escribe: `python --version`

### Paso 1 — Descargar el proyecto

Si usas GitHub:
```bash
git clone https://github.com/TU_USUARIO/cbb-predictor.git
cd cbb-predictor
```
O simplemente descarga la carpeta y entra en ella desde la terminal.

### Paso 2 — (Recomendado) Crear un entorno virtual

Un "entorno virtual" mantiene las librerías de este proyecto separadas
del resto de tu computadora.

```bash
# Crear el entorno
python -m venv venv

# Activarlo:
#   En Windows:
venv\Scripts\activate
#   En Mac / Linux:
source venv/bin/activate
```

### Paso 3 — Instalar las librerías

```bash
pip install -r requirements.txt
```

### Paso 4 — Entrenar el modelo

```bash
python src/train.py
```
Esto crea el archivo `src/modelo_entrenado.pkl` y muestra la precisión
del modelo y un ejemplo de predicción.

✅ **¡Listo!** Ya puedes usar la API o el dashboard.

---

## 🔮 Cómo hacer predicciones

Tienes **dos opciones**. Puedes usar la que prefieras.

### Opción A — Dashboard web (la más fácil y visual) 🌟

```bash
streamlit run dashboard/app.py
```
Se abrirá una página web (normalmente en `http://localhost:8501`).
Ahí eliges dos equipos, haces clic en **"Predecir ganador"** y ves
las probabilidades y gráficas de la temporada.

### Opción B — API REST (para programadores)

En una terminal, inicia la API:
```bash
python api/app.py
```
La API quedará disponible en `http://localhost:5000`.

Ejemplos de uso (en otra terminal o en el navegador):

```bash
# Ver la lista de equipos disponibles
curl http://localhost:5000/equipos

# Predecir un partido (Duke en casa vs Kansas)
curl "http://localhost:5000/predecir?local=Duke&visitante=Kansas"
```

Respuesta de ejemplo:
```json
{
  "equipo_local": "Duke",
  "equipo_visitante": "Kansas",
  "prob_gana_local": 0.58,
  "prob_gana_visitante": 0.42,
  "ganador_predicho": "Duke",
  "confianza": 0.58
}
```

También puedes enviar los datos como JSON con POST:
```bash
curl -X POST http://localhost:5000/predecir \
     -H "Content-Type: application/json" \
     -d '{"local": "Duke", "visitante": "Kansas"}'
```

---

## 🧠 ¿Cómo funciona el modelo? (explicación sencilla)

El modelo es una **Regresión Logística**, uno de los algoritmos más
simples de Machine Learning. Para cada partido mira la **diferencia**
entre el equipo local y el visitante en:

- 📈 **Puntos anotados** recientes (promedio de los últimos 5 partidos).
- 🛡️ **Puntos recibidos** recientes (qué tan buena es su defensa).
- 🔥 **Racha** de victorias reciente.
- 🏆 **Porcentaje de victorias** en la temporada.

Estas "pistas" se llaman **features** (características). El modelo aprende
de los partidos pasados qué combinación suele llevar a una victoria del
equipo local, e incluye también la **ventaja de jugar en casa**.

> ℹ️ Como los datos son **simulados** (de ejemplo), la precisión es
> moderada. Si conectas datos reales de una temporada, el modelo mejora.

---

## ❓ Problemas comunes

| Problema | Solución |
|----------|----------|
| `No se encontró el modelo...` | Ejecuta primero `python src/train.py` |
| `command not found: python` | Prueba con `python3` en vez de `python` |
| `ModuleNotFoundError` | Ejecuta `pip install -r requirements.txt` |
| El puerto 5000 está ocupado | Cambia el puerto en `api/app.py` (línea final) |

---

## 🔄 Cambiar los datos por los tuyos

Reemplaza `data/cbb_games.csv` con tus propios datos, manteniendo
**las mismas columnas**:

```
fecha, equipo_local, equipo_visitante, puntos_local, puntos_visitante, gano_local
```

Luego vuelve a entrenar:
```bash
python src/train.py
```

---

## 📝 Notas finales

- Proyecto pensado para **aprender**, no para apostar dinero. 🙂
- Todo el código está comentado en español para que lo entiendas y
  lo modifiques a tu gusto.

¡Disfrútalo y sigue aprendiendo! 🚀
