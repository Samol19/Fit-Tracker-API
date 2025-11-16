# Modelo de Sentadillas (Squat)

## Información General
- **Tipo de Modelo**: Random Forest Classifier
- **Accuracy**: 100% (14/14 muestras test correctas)
- **Total Features**: 20 (4 métricas × 5 estadísticas)
- **Clases**: 4 categorías de error
- **Técnica de Balanceo**: class_weight='balanced'
- **Ángulo de Cámara**: Solo diagonal (filtrado '_diag')

## Archivos del Modelo
```
API/model/
├── squat_classifier_model.pkl  # Modelo entrenado
├── squat_label_encoder.pkl     # Codificador de etiquetas
└── squat_scaler.pkl            # StandardScaler para normalización
```

## Clases de Salida
1. **squat_correcto**: Ejecución correcta
2. **squat_poca_profundidad**: Profundidad insuficiente (rodillas no bajan <90°)
3. **squat_espalda_arqueada**: Espalda no mantiene alineación recta
4. **squat_valgo_rodilla**: Rodillas colapsan hacia adentro (valgo)

## Features de Entrada (20 features)

### Métricas Base (4)
Calculadas durante todo el movimiento de sentadilla completa:

1. **avg_knee_angle**: Ángulo promedio de las rodillas (cadera-rodilla-tobillo)
   - Indica profundidad del descenso
   - Valores bajos = mayor profundidad
   
2. **avg_hip_angle**: Ángulo promedio de la cadera (hombro-cadera-rodilla)
   - Evalúa flexión de cadera
   - Relacionado con postura de espalda

3. **knee_distance**: Distancia horizontal entre rodillas
   - Detecta valgo de rodilla (colapso interno)
   - Valores bajos = rodillas juntas (error)

4. **hip_shoulder_distance**: Distancia vertical cadera-hombro
   - Evalúa inclinación del torso
   - Relacionado con alineación de espalda

### Estadísticas Calculadas (×5 por métrica)
Para cada una de las 4 métricas base:
- `_mean`: Promedio durante el movimiento
- `_std`: Desviación estándar (variabilidad)
- `_min`: Valor mínimo alcanzado
- `_max`: Valor máximo alcanzado
- `_range`: Rango (max - min)

**Total**: 4 métricas × 5 estadísticas = **20 features**

## Importancia de Features

### Top 10 Features (por importancia)
| Rank | Feature | Importancia | Interpretación |
|------|---------|-------------|----------------|
| 1 | knee_distance_min | 9.2% | Mínima separación de rodillas (valgo) |
| 2 | hip_shoulder_distance_mean | 9.2% | Inclinación promedio del torso |
| 3 | knee_distance_mean | 8.8% | Separación promedio de rodillas |
| 4 | avg_knee_angle_min | 8.7% | Profundidad máxima del descenso |
| 5 | avg_hip_angle_std | 8.7% | Variabilidad de flexión de cadera |
| 6 | avg_knee_angle_range | 8.6% | Rango de movimiento de rodilla |
| 7 | avg_hip_angle_range | 8.5% | Rango de flexión de cadera |
| 8 | knee_distance_max | 7.9% | Separación máxima de rodillas |
| 9 | avg_knee_angle_mean | 7.8% | Ángulo promedio de rodilla |
| 10 | hip_shoulder_distance_std | 7.0% | Variabilidad de torso |

**Las 10 features más importantes representan 84.4% de la capacidad predictiva del modelo.**

## Detección de Repeticiones

### Método: Máquina de Estados basada en Ángulo de Rodilla

**Estados del Movimiento**:
- `ARRIBA`: Posición inicial/final (rodillas extendidas)
- `BAJANDO`: Descenso activo
- `ABAJO`: Posición de máxima flexión
- `SUBIENDO`: Ascenso activo

### Umbrales de Transición
```python
KNEE_ANGLE_DOWN_THRESHOLD = 160°   # Inicio del descenso
KNEE_ANGLE_UP_THRESHOLD = 170°     # Completar la subida
KNEE_ANGLE_BOTTOM_THRESHOLD = 90°  # Posición baja (profundidad)

ESTADO = "ARRIBA"  # Estado inicial

# Transiciones
if ESTADO == "ARRIBA" and knee_angle < KNEE_ANGLE_DOWN_THRESHOLD:
    ESTADO = "BAJANDO"
    
elif ESTADO == "BAJANDO" and knee_angle < KNEE_ANGLE_BOTTOM_THRESHOLD:
    ESTADO = "ABAJO"
    # Aquí se guarda el frame del punto más bajo
    
elif ESTADO == "ABAJO" and knee_angle > KNEE_ANGLE_DOWN_THRESHOLD:
    ESTADO = "SUBIENDO"
    
elif ESTADO == "SUBIENDO" and knee_angle > KNEE_ANGLE_UP_THRESHOLD:
    # REPETICIÓN COMPLETA → Clasificar
    ESTADO = "ARRIBA"
    rep_count += 1
```

### Ventana de Análisis
- Se guardan **todos los frames** desde inicio del descenso hasta final de la subida
- Al completar la repetición, se calculan las 20 features estadísticas sobre esta ventana completa
- Ventana típica: 60-120 frames (2-4 segundos)

## Configuración de Sensibilidad

Multiplicadores de probabilidad ajustados empíricamente:

```python
SENSIBILIDAD_CLASE = {
    'squat_correcto': 1.0,              # Normal
    'squat_poca_profundidad': 4.0,      # MUY alta sensibilidad (error común)
    'squat_espalda_arqueada': 1.2,      # Ligeramente más sensible
    'squat_valgo_rodilla': 2.5,         # Alta sensibilidad
}
```

**Justificación de valores altos**:
- `squat_poca_profundidad (4.0)`: Error muy frecuente, modelo tiende a sub-detectarlo
- `squat_valgo_rodilla (2.5)`: Error crítico que requiere corrección inmediata
- `squat_espalda_arqueada (1.2)`: Ajuste fino para balance

**Cómo funciona**:
```python
# Multiplicar probabilidades predichas
adjusted_probs = probs * sensitivity_multipliers
# Re-normalizar
adjusted_probs /= adjusted_probs.sum()
# Predicción final
prediction = class_with_highest_adjusted_prob
```

## Requisitos de MediaPipe

### Landmarks Necesarios
```python
LEFT_SHOULDER, RIGHT_SHOULDER  # Para torso
LEFT_HIP, RIGHT_HIP            # Para ángulos y distancia vertical
LEFT_KNEE, RIGHT_KNEE          # Para profundidad y valgo
LEFT_ANKLE, RIGHT_ANKLE        # Para ángulo de rodilla
```

### Ángulo de Cámara
- **SOLO Diagonal**: Modelo entrenado únicamente con vistas diagonales
- **NO frontal**: Vistas frontales **no** están incluidas en el entrenamiento
- **Altura**: Cámara a nivel de cadera, capturando desde hombros hasta pies
- **Distancia**: 2.5-3.5 metros para cuerpo completo visible

### ⚠️ Importante: Limitación de Ángulo
El modelo **NO** funcionará correctamente con cámara frontal. Solo usar posiciones diagonales (≈45°).

## Preprocesamiento

### 1. Extracción de Features por Frame
```python
def extract_frame_metrics(landmarks):
    # Promedios de ambos lados
    avg_knee_angle = (knee_angle_left + knee_angle_right) / 2
    avg_hip_angle = (hip_angle_left + hip_angle_right) / 2
    
    # Distancia horizontal entre rodillas
    knee_distance = abs(left_knee.x - right_knee.x)
    
    # Distancia vertical cadera-hombro
    hip_shoulder_distance = abs(avg_hip.y - avg_shoulder.y)
    
    return {
        'avg_knee_angle': avg_knee_angle,
        'avg_hip_angle': avg_hip_angle,
        'knee_distance': knee_distance,
        'hip_shoulder_distance': hip_shoulder_distance
    }
```

### 2. Detección de Repetición Completa
```python
# Acumular frames durante toda la sentadilla
frame_buffer = []

# Máquina de estados
if squat_completed(state_machine):
    # Calcular estadísticas sobre todos los frames del movimiento
    features = calculate_statistics(frame_buffer)
```

### 3. Cálculo de Estadísticas
```python
def calculate_statistics(frames_data):
    features = {}
    for metric in ['avg_knee_angle', 'avg_hip_angle', 'knee_distance', 'hip_shoulder_distance']:
        values = [frame[metric] for frame in frames_data]
        
        features[f'{metric}_mean'] = np.mean(values)
        features[f'{metric}_std'] = np.std(values)
        features[f'{metric}_min'] = np.min(values)
        features[f'{metric}_max'] = np.max(values)
        features[f'{metric}_range'] = np.max(values) - np.min(values)
    
    return features  # 20 features
```

### 4. Normalización
```python
# Aplicar StandardScaler entrenado
features_array = np.array([list(features.values())])
features_scaled = scaler.transform(features_array)
```

### 5. Predicción con Sensibilidad
```python
# Obtener probabilidades base
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
import mediapipe as mp

# Cargar modelo
model = joblib.load('API/model/squat_classifier_model.pkl')
le = joblib.load('API/model/squat_label_encoder.pkl')
scaler = joblib.load('API/model/squat_scaler.pkl')

# Estado de la máquina
ESTADO = "ARRIBA"
frame_buffer = []
rep_count = 0

def process_frame(landmarks):
    global ESTADO, frame_buffer, rep_count
    
    # 1. Extraer métricas del frame actual
    metrics = extract_frame_metrics(landmarks)
    knee_angle = metrics['avg_knee_angle']
    
    # 2. Máquina de estados
    if ESTADO == "ARRIBA" and knee_angle < 160:
        ESTADO = "BAJANDO"
        frame_buffer = [metrics]  # Iniciar nueva repetición
        
    elif ESTADO == "BAJANDO":
        frame_buffer.append(metrics)
        if knee_angle < 90:
            ESTADO = "ABAJO"
            
    elif ESTADO == "ABAJO":
        frame_buffer.append(metrics)
        if knee_angle > 160:
            ESTADO = "SUBIENDO"
            
    elif ESTADO == "SUBIENDO":
        frame_buffer.append(metrics)
        if knee_angle > 170:
            # REPETICIÓN COMPLETA
            ESTADO = "ARRIBA"
            rep_count += 1
            
            # 3. Calcular features estadísticas
            features = calculate_statistics(frame_buffer)
            features_array = np.array([list(features.values())])
            features_scaled = scaler.transform(features_array)
            
            # 4. Clasificar con sensibilidad
            probs = model.predict_proba(features_scaled)[0]
            sens = np.array([SENSIBILIDAD_CLASE[c] for c in le.classes_])
            adjusted_probs = probs * sens
            adjusted_probs /= adjusted_probs.sum()
            
            prediction = le.classes_[np.argmax(adjusted_probs)]
            confidence = adjusted_probs.max()
            
            return {
                'repetition_complete': True,
                'rep_number': rep_count,
                'class': prediction,
                'confidence': float(confidence),
                'frame_count': len(frame_buffer)
            }
    
    # Estado intermedio
    return {
        'repetition_complete': False,
        'current_state': ESTADO,
        'knee_angle': float(knee_angle)
    }
```

## Métricas de Rendimiento

### Accuracy General: 100%

### Por Clase (Test Set):
| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| squat_correcto | 1.00 | 1.00 | 1.00 | 5 |
| squat_poca_profundidad | 1.00 | 1.00 | 1.00 | 3 |
| squat_espalda_arqueada | 1.00 | 1.00 | 1.00 | 4 |
| squat_valgo_rodilla | 1.00 | 1.00 | 1.00 | 2 |

### Dataset:
- **Total repeticiones**: 69 (solo cámara diagonal)
- **Train/Test split**: 80/20 (55 train, 14 test)
- **Videos por clase**: 
  - squat_correcto: 26 reps (diag)
  - squat_poca_profundidad: 14 reps (diag)
  - squat_espalda_arqueada: 17 reps (diag)
  - squat_valgo_rodilla: 12 reps (diag)
- **Balance**: class_weight='balanced' en RandomForest

### Matriz de Confusión (Test):
```
Todas las predicciones correctas:
                    Predicho
                    corr  prof  espa  valg
Real correcto        5     0     0     0
     poca_prof       0     3     0     0
     espalda_arq     0     0     4     0
     valgo_rod       0     0     0     2
```

## Notas Importantes

1. **Solo Diagonal**: El modelo **SOLO** funciona con cámara diagonal (≈45°), no entrenado con vistas frontales
2. **Sensibilidad Alta**: `squat_poca_profundidad` tiene sensibilidad 4x para compensar sub-detección
3. **Movimiento Completo**: Requiere sentadilla completa (bajada + subida) para clasificar, no evalúa frames individuales
4. **100% Accuracy**: Logrado gracias a:
   - 20 features estadísticas (5 por métrica)
   - Solo ángulos diagonales (mayor consistencia)
   - class_weight='balanced'
5. **Estado Real-time**: Usa máquina de estados para robustez, no solo umbrales simples

## Limitaciones

- **Ángulo de cámara**: Solo diagonal, NO frontal
- Requiere sentadilla completa visible (inicio a fin)
- Necesita buena iluminación para MediaPipe
- Distancia óptima: 2.5-3.5 metros
- No evalúa velocidad de ejecución (solo forma)
- Modelo específico para dataset con 3 reps por video

## Configuración Recomendada para Producción

```python
# Umbrales conservadores para evitar falsos positivos
KNEE_ANGLE_DOWN_THRESHOLD = 165°   # Más estricto
KNEE_ANGLE_UP_THRESHOLD = 175°     # Más estricto
KNEE_ANGLE_BOTTOM_THRESHOLD = 95°  # Más permisivo

# Sensibilidad ajustada después de pruebas con usuarios reales
SENSIBILIDAD_CLASE = {
    'squat_correcto': 1.0,
    'squat_poca_profundidad': 3.5,    # Reducir ligeramente si muchos FP
    'squat_espalda_arqueada': 1.2,
    'squat_valgo_rodilla': 2.5,
}

# Mínimo de frames para clasificación válida
MIN_FRAMES_PER_REP = 45  # ~1.5 segundos a 30fps
```
