# API de Inferencia en Tiempo Real (Random Forest)

API FastAPI + WebSocket para predicción de poses/plank usando RandomForest.

## Estructura
- `main.py`: Servidor FastAPI con endpoint WebSocket `/ws/predict`
- `model/`: Carpeta donde debes colocar los archivos:
    - `plank_classifier_model.pkl`
    - `plank_scaler.pkl`
    - `plank_label_encoder.pkl`
- `requirements.txt`: Dependencias Python
- `Dockerfile`: Listo para Railway

## Uso local
1. Copia los archivos del modelo y scaler a `API/model/`
2. Instala dependencias:
   ```
   pip install -r requirements.txt
   ```
3. Ejecuta el servidor:
   ```
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

## Despliegue en Railway
1. Sube todo el contenido de la carpeta `API` a tu proyecto Railway.
2. Asegúrate de que los archivos del modelo estén en `model/`.
3. Railway detectará el `Dockerfile` y expondrá el puerto 8000.

## Ejemplo de mensaje WebSocket
Envío:
```json
{"features": [0.1, 0.2, ..., 0.25]}
```
Respuesta:
```json
{"proba": [0.1, 0.7, 0.2, 0.0], "pred": "plank_correcto", "pred_idx": 2}
```
