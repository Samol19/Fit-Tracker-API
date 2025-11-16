# Modelo de Plancha (Plank)

## Información General
- **Tipo de Modelo**: Random Forest Classifier
- **Accuracy**: ~85-90% (estimado del classification report)
- **Total Features**: Variable (calculadas en tiempo real)
- **Clases**: 4 categorías de error
- **Técnica de Balanceo**: class_weight='balanced'
- **Preprocesamiento**: StandardScaler

## Archivos del Modelo
```
API/model/
├── plank_classifier_model.pkl  # Modelo entrenado
├── plank_label_encoder.pkl     # Codificador de etiquetas
└── plank_scaler.pkl            # StandardScaler para normalización
```

## Clases de Salida
1. **plank_correcto**: Ejecución correcta (cuerpo alineado)
2. **plank_cadera_caida**: Cadera caída (hacia el suelo)
3. **plank_codos_abiertos**: Codos muy separados (>hombros)
4. **plank_pelvis_levantada**: Pelvis/cadera elevada (no alineada)

## Features de Entrada

El modelo de plank usa features calculadas en tiempo real basadas en:

### Métricas Principales
1. **body_angle**: Ángulo hombro-cadera-tobillo
   - Indica alineación corporal general
   - Valor ideal: ≈180° (línea recta)
   - Detecta cadera caída (<180°) o pelvis levantada (varía)

2. **elbow_angle**: Ángulo hombro-codo-muñeca
   - Detecta flexión excesiva de codos
   - Valor ideal: ≈90° (codos en ángulo recto)

3. **shoulder_elbow_distance**: Distancia horizontal hombros-codos
   - Detecta codos muy abiertos
   - Comparar con ancho de hombros

4. **hip_height**: Altura relativa de la cadera
   - Posición vertical de cadera vs hombros
   - Detecta cadera caída o pelvis elevada

5. **shoulder_alignment**: Alineación de hombros
   - Evalúa simetría del cuerpo
   - Detecta rotación o inclinación

### Características del Dataset
- Las features exactas se generan dinámicamente en `build_training_dataset.py`
- Ventanas temporales de 1 segundo (≈30 frames)
- Estadísticas agregadas: promedios, desviaciones, mínimos, máximos

## Detección en Tiempo Real

### Método: Buffer Temporal con Clasificación Continua

**Diferencia clave**: A diferencia de pushup (repeticiones) y squat (estado), plank es **postura estática**.

### Proceso de Evaluación
```python
BUFFER_SIZE_SECONDS = 1  # Ventana de evaluación
FPS_ESTIMADO = 30
BUFFER_FRAME_SIZE = 30  # frames

# Acumular frames
feature_buffer = []

# Por cada frame
while True:
    features = extract_features(landmarks)
    feature_buffer.append(features)
    
    # Mantener solo últimos 30 frames
    if len(feature_buffer) > BUFFER_FRAME_SIZE:
        feature_buffer.pop(0)
    
    # Si buffer lleno, clasificar
    if len(feature_buffer) == BUFFER_FRAME_SIZE:
        aggregated_features = aggregate(feature_buffer)
        prediction = classify(aggregated_features)
```

### Sin Conteo de Repeticiones
- No hay máquina de estados
- No se detectan picos
- Clasificación **continua** de la postura actual
- Feedback inmediato (cada 1 segundo)

## Configuración de Sensibilidad

```python
SENSIBILIDAD_CLASE = {
    'plank_cadera_caida': 1.0,        # Normal
    'plank_codos_abiertos': 0.5,      # Menos sensible (error menos crítico)
    'plank_correcto': 1.3,            # Más sensible (favorecer correcto)
    'plank_pelvis_levantada': 1.0,    # Normal
}
```

**Estrategia de sensibilidad**:
- `plank_correcto (1.3)`: **Favorece** detección de postura correcta para feedback positivo
- `plank_codos_abiertos (0.5)`: **Reduce** falsos positivos de este error menos crítico
- Otros errores: Sensibilidad normal para balance

## Requisitos de MediaPipe

### Landmarks Necesarios
```python
LEFT_SHOULDER, RIGHT_SHOULDER  # Para alineación y ancho
LEFT_ELBOW, RIGHT_ELBOW        # Para ángulo de codos
LEFT_WRIST, RIGHT_WRIST        # Para ángulo de codos
LEFT_HIP, RIGHT_HIP            # Para ángulo corporal y altura
LEFT_ANKLE, RIGHT_ANKLE        # Para ángulo corporal
```

### Ángulo de Cámara
- **Recomendado**: Lateral (90°)
- **Funciona**: Diagonal (45-135°)
- **NO**: Frontal (vista de frente no permite evaluar alineación)
- **Altura**: Cámara a nivel del cuerpo (ni muy alta ni muy baja)
- **Distancia**: 2-3 metros, cuerpo completo visible

## Preprocesamiento

### 1. Extracción de Features por Frame
```python
def extract_plank_features(landmarks):
    # Promedios de landmarks bilaterales
    avg_shoulder = average(left_shoulder, right_shoulder)
    avg_elbow = average(left_elbow, right_elbow)
    avg_hip = average(left_hip, right_hip)
    avg_ankle = average(left_ankle, right_ankle)
    
    # Calcular métricas
    body_angle = calculate_angle(avg_shoulder, avg_hip, avg_ankle)
    elbow_angle_left = calculate_angle(left_shoulder, left_elbow, left_wrist)
    elbow_angle_right = calculate_angle(right_shoulder, right_elbow, right_wrist)
    elbow_angle = (elbow_angle_left + elbow_angle_right) / 2
    
    shoulder_width = distance(left_shoulder, right_shoulder)
    elbow_width = distance(left_elbow, right_elbow)
    shoulder_elbow_distance = abs(elbow_width - shoulder_width)
    
    hip_height = avg_hip.y  # Coordenada Y (normalizada 0-1)
    
    return {
        'body_angle': body_angle,
        'elbow_angle': elbow_angle,
        'shoulder_elbow_distance': shoulder_elbow_distance,
        'hip_height': hip_height,
        # ... otras features
    }
```

### 2. Agregación de Buffer Temporal
```python
def aggregate_features(feature_buffer):
    # Buffer de 30 frames (1 segundo)
    aggregated = {}
    
    for feature_name in ['body_angle', 'elbow_angle', 'hip_height', ...]:
        values = [frame[feature_name] for frame in feature_buffer]
        
        aggregated[f'{feature_name}_mean'] = np.mean(values)
        aggregated[f'{feature_name}_std'] = np.std(values)
        aggregated[f'{feature_name}_min'] = np.min(values)
        aggregated[f'{feature_name}_max'] = np.max(values)
        # Posiblemente más estadísticas
    
    return aggregated
```

### 3. Normalización
```python
# Aplicar StandardScaler
features_array = np.array([list(aggregated.values())])
features_scaled = scaler.transform(features_array)
```

### 4. Predicción con Sensibilidad
```python
# Probabilidades base
probs = model.predict_proba(features_scaled)[0]

# Aplicar multiplicadores
sens = np.array([SENSIBILIDAD_CLASE[clase] for clase in le.classes_])
adjusted_probs = probs * sens
adjusted_probs /= adjusted_probs.sum()

# Predicción final
prediction_idx = np.argmax(adjusted_probs)
prediction = le.classes_[prediction_idx]
confidence = adjusted_probs[prediction_idx]
```

## Ejemplo de Uso con WebSocket

```python
import joblib
import numpy as np
from collections import deque

# Cargar modelo
model = joblib.load('API/model/plank_classifier_model.pkl')
le = joblib.load('API/model/plank_label_encoder.pkl')
scaler = joblib.load('API/model/plank_scaler.pkl')

# Buffer circular
BUFFER_SIZE = 30  # frames
feature_buffer = deque(maxlen=BUFFER_SIZE)

def process_frame(landmarks):
    # 1. Extraer features del frame actual
    features = extract_plank_features(landmarks)
    
    # 2. Agregar a buffer
    feature_buffer.append(features)
    
    # 3. Si buffer lleno, clasificar
    if len(feature_buffer) == BUFFER_SIZE:
        # Agregar features del buffer
        aggregated = aggregate_features(list(feature_buffer))
        features_array = np.array([list(aggregated.values())])
        features_scaled = scaler.transform(features_array)
        
        # Clasificar con sensibilidad
        probs = model.predict_proba(features_scaled)[0]
        sens = np.array([SENSIBILIDAD_CLASE[c] for c in le.classes_])
        adjusted_probs = probs * sens
        adjusted_probs /= adjusted_probs.sum()
        
        prediction = le.classes_[np.argmax(adjusted_probs)]
        confidence = adjusted_probs.max()
        
        return {
            'ready': True,
            'class': prediction,
            'confidence': float(confidence),
            'body_angle': float(features['body_angle']),
            'elbow_angle': float(features['elbow_angle'])
        }
    else:
        # Buffer aún llenándose
        return {
            'ready': False,
            'buffer_progress': len(feature_buffer) / BUFFER_SIZE
        }
```

## Métricas de Rendimiento

### Accuracy Estimado: 85-90%

### Características del Modelo
- **Tipo**: RandomForestClassifier
- **Parámetros**:
  - `n_estimators=100`
  - `random_state=42`
  - `class_weight='balanced'`
- **Train/Test Split**: 75/25

### Dataset
- Videos de plancha con 4 categorías de error
- Segmentos temporales extraídos de cada video
- Features agregadas por ventanas de tiempo
- Normalización con StandardScaler

## Comparación con Otros Modelos

| Aspecto | Plank | Pushup | Squat |
|---------|-------|--------|-------|
| **Tipo de Ejercicio** | Postura estática | Repeticiones dinámicas | Repeticiones dinámicas |
| **Método Detección** | Buffer temporal | Picos en señal | Máquina de estados |
| **Features** | Agregadas de 1s | 30 estadísticas | 20 estadísticas |
| **Clasificación** | Continua | Por repetición | Por repetición |
| **Ángulo Cámara** | Lateral/Diagonal | Cualquiera | Solo diagonal |
| **Output** | Feedback instantáneo | Conteo + clase | Conteo + clase |

## Notas Importantes

1. **Postura Estática**: No cuenta repeticiones, evalúa postura constantemente
2. **Feedback Continuo**: Clasificación cada 1 segundo (buffer completo)
3. **Sensibilidad Balanceada**: Favorece `plank_correcto` (1.3x) para motivación positiva
4. **Cámara Lateral**: Esencial para evaluar alineación corporal
5. **Buffer Simple**: Sin complicaciones de detección de picos o estados

## Limitaciones

- Requiere vista lateral clara (no frontal)
- Sensible a ángulo de cámara (debe ser perpendicular al cuerpo)
- No evalúa tiempo de aguante (solo postura)
- Necesita cuerpo completo visible
- Buffer de 1 segundo introduce latencia mínima

## Configuración Recomendada para Producción

```python
# Buffer temporal
BUFFER_SIZE_SECONDS = 1.0  # Mantener en 1 segundo
FPS_TARGET = 30            # FPS esperado

# Sensibilidad ajustada para producción
SENSIBILIDAD_CLASE = {
    'plank_correcto': 1.3,            # Favorece feedback positivo
    'plank_cadera_caida': 1.0,        # Error crítico, sensibilidad normal
    'plank_codos_abiertos': 0.5,      # Menos crítico, reducir FP
    'plank_pelvis_levantada': 1.0,    # Error moderado, sensibilidad normal
}

# Confianza mínima para mostrar feedback
MIN_CONFIDENCE = 0.60  # Solo mostrar si confianza > 60%

# Suavizado de predicciones (opcional)
PREDICTION_SMOOTHING = 3  # Mayoría de últimas 3 predicciones
```

## Uso con WebSocket - Flujo Completo

```python
import asyncio
import websockets
import json

async def plank_feedback_handler(websocket, path):
    feature_buffer = deque(maxlen=30)
    
    async for message in websocket:
        # Recibir landmarks del cliente
        landmarks = json.loads(message)
        
        # Procesar frame
        result = process_frame(landmarks)
        
        # Enviar feedback
        if result['ready']:
            await websocket.send(json.dumps({
                'type': 'classification',
                'class': result['class'],
                'confidence': result['confidence'],
                'metrics': {
                    'body_angle': result['body_angle'],
                    'elbow_angle': result['elbow_angle']
                }
            }))
        else:
            await websocket.send(json.dumps({
                'type': 'buffering',
                'progress': result['buffer_progress']
            }))

# Iniciar servidor WebSocket
start_server = websockets.serve(plank_feedback_handler, "localhost", 8765)
asyncio.get_event_loop().run_until_complete(start_server)
```

## Features Exactas del Modelo

Para obtener la lista exacta de features utilizadas por el modelo:

```python
import joblib
import pandas as pd

# Cargar scaler para ver las features esperadas
scaler = joblib.load('API/model/plank_scaler.pkl')
print(f"Total features esperadas: {scaler.n_features_in_}")

# Cargar dataset de entrenamiento para ver nombres de columnas
df = pd.read_csv('FIT_TRACKER/plank_model/plank_training_dataset.csv')
features = [col for col in df.columns if col not in ['class', 'video_segment']]
print("Features del modelo:")
for i, feat in enumerate(features, 1):
    print(f"{i}. {feat}")
```

Este comando revelará las features exactas usadas durante el entrenamiento.
