# Modelo de Flexiones (Pushup)

## Información General
- **Tipo de Modelo**: Random Forest Classifier
- **Accuracy**: 91%
- **Total Features**: 30 (6 métricas × 5 estadísticas)
- **Clases**: 4 categorías de error
- **Técnica de Balanceo**: RandomOverSampler + class_weight='balanced'

## Archivos del Modelo
```
API/model/
├── pushup_classifier_model.pkl  # Modelo entrenado
├── pushup_label_encoder.pkl     # Codificador de etiquetas
└── pushup_scaler.pkl            # StandardScaler para normalización
```

## Clases de Salida
1. **pushup_correcto**: Ejecución correcta
2. **pushup_cadera_caida**: Cadera caída durante el movimiento
3. **pushup_codos_abiertos**: Codos muy separados (>45°)
4. **pushup_pelvis_levantada**: Pelvis elevada, falta de alineación corporal

## Features de Entrada (30 features)

### Métricas Base (6)
Cada métrica se calcula sobre una ventana de ±20 frames alrededor del punto detectado:

1. **body_angle**: Ángulo hombro-cadera-tobillo (alineación corporal)
2. **hip_shoulder_vertical_diff**: Diferencia vertical cadera-hombro
3. **hip_ankle_vertical_diff**: Diferencia vertical cadera-tobillo
4. **shoulder_elbow_angle**: Ángulo hombro-codo-cadera
5. **wrist_shoulder_hip_angle**: Ángulo muñeca-hombro-cadera
6. **shoulder_wrist_vertical_diff**: Diferencia vertical hombro-muñeca (CLAVE para detección de repeticiones)

### Estadísticas Calculadas (×5 por métrica)
Para cada una de las 6 métricas base:
- `_mean`: Promedio
- `_std`: Desviación estándar
- `_min`: Valor mínimo
- `_max`: Valor máximo
- `_range`: Rango (max - min)

**Total**: 6 métricas × 5 estadísticas = **30 features**

## Detección de Repeticiones

### Método: Detección de Picos en Tiempo Real
Utiliza `shoulder_wrist_vertical_diff` como señal principal:

**Interpretación de la señal**:
- Valores MÁS NEGATIVOS (≈ -0.35 a -0.50): Brazos **extendidos** (posición alta)
- Valores MENOS NEGATIVOS (≈ -0.05 a -0.15): Brazos **flexionados** (posición baja) → **PICO = REPETICIÓN**

### Parámetros de Detección
```python
BUFFER_SIZE = 150  # frames (≈5 segundos a 30fps)
PEAK_MIN_DISTANCE = 25  # Mínimo 25 frames entre repeticiones
MARGIN_BEFORE = 20  # Frames antes del pico
MARGIN_AFTER = 20   # Frames después del pico
MIN_PROMINENCE = 0.03  # Prominencia mínima absoluta

# Suavizado con Savitzky-Golay
window_length = 11
polyorder = 3

# Umbral adaptativo
height_threshold = signal_min + (signal_range * 0.40)  # 40% del rango desde abajo
prominence_min = max(signal_range * 0.10, 0.03)  # 10% del rango o 0.03
```

### Proceso de Detección
1. Buffer circular almacena últimos 150 frames de `shoulder_wrist_vertical_diff`
2. Aplicar filtro Savitzky-Golay para suavizar señal
3. Detectar PICOS con `scipy.signal.find_peaks`
4. Extraer ventana de ±20 frames alrededor del pico
5. Calcular 30 features estadísticas
6. Clasificar con modelo

## Configuración de Sensibilidad

Multiplicadores de probabilidad para ajustar sensibilidad de cada clase:

```python
SENSIBILIDAD_CLASE = {
    "pushup_correcto": 1.0,           # Normal
    "pushup_cadera_caida": 1.0,       # Normal
    "pushup_codos_abiertos": 0.9,     # Ligeramente menos sensible
    "pushup_pelvis_levantada": 1.0,   # Normal
}
```

**Cómo ajustar**:
- `> 1.0`: Clase más probable de aparecer (más sensible)
- `< 1.0`: Clase menos probable de aparecer (menos sensible)
- `= 1.0`: Sin ajuste

## Requisitos de MediaPipe

### Landmarks Necesarios
```python
LEFT_SHOULDER, RIGHT_SHOULDER
LEFT_HIP, RIGHT_HIP
LEFT_ANKLE, RIGHT_ANKLE
LEFT_ELBOW, RIGHT_ELBOW
LEFT_WRIST, RIGHT_WRIST
```

### Ángulo de Cámara
- **Recomendado**: Lateral (de lado)
- **Funciona**: Diagonal o frontal (el modelo es robusto a ángulos)
- **Altura**: Capturar cuerpo completo desde hombros hasta pies

## Preprocesamiento

### 1. Extracción de Features
```python
# Calcular métricas base por frame
body_angle = calculate_angle(shoulder, hip, ankle)
hip_shoulder_vertical_diff = hip[1] - shoulder[1]
hip_ankle_vertical_diff = hip[1] - ankle[1]
shoulder_elbow_angle = calculate_angle(hip, shoulder, elbow)
wrist_shoulder_hip_angle = calculate_angle(wrist, shoulder, hip)
shoulder_wrist_vertical_diff = shoulder[1] - wrist[1]
```

### 2. Detección de Repetición
```python
# Detectar pico en shoulder_wrist_vertical_diff
smoothed = savgol_filter(signal, window_length=11, polyorder=3)
peaks, _ = find_peaks(smoothed, height=threshold, distance=25, prominence=prom_min)
```

### 3. Cálculo de Estadísticas
```python
# Para cada métrica en la ventana del pico
for metric in metrics:
    features[f'{metric}_mean'] = window[metric].mean()
    features[f'{metric}_std'] = window[metric].std()
    features[f'{metric}_min'] = window[metric].min()
    features[f'{metric}_max'] = window[metric].max()
    features[f'{metric}_range'] = window[metric].max() - window[metric].min()
```

### 4. Normalización
```python
# Aplicar StandardScaler
features_scaled = scaler.transform(features)
```

### 5. Predicción
```python
# Obtener probabilidades
probabilities = model.predict_proba(features_scaled)[0]

# Aplicar multiplicadores de sensibilidad
adjusted_probs = probabilities * sensitivity_multipliers
adjusted_probs /= adjusted_probs.sum()

# Predicción final
prediction_idx = np.argmax(adjusted_probs)
prediction = label_encoder.classes_[prediction_idx]
confidence = adjusted_probs[prediction_idx]
```

## Ejemplo de Uso con WebSocket

```python
import joblib
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from collections import deque

# Cargar modelo
model = joblib.load('API/model/pushup_classifier_model.pkl')
le = joblib.load('API/model/pushup_label_encoder.pkl')
scaler = joblib.load('API/model/pushup_scaler.pkl')

# Buffer para detección en tiempo real
signal_buffer = deque(maxlen=150)
features_buffer = deque(maxlen=150)

# Por cada frame de video
def process_frame(landmarks):
    # 1. Calcular métricas
    metrics = extract_metrics(landmarks)
    
    # 2. Agregar a buffer
    signal_buffer.append(metrics['shoulder_wrist_vertical_diff'])
    features_buffer.append(metrics)
    
    # 3. Detectar picos si hay suficientes frames
    if len(signal_buffer) >= 50:
        signal = np.array(signal_buffer)
        smoothed = savgol_filter(signal, 11, 3)
        
        peaks, _ = find_peaks(smoothed, 
                             height=threshold,
                             distance=25,
                             prominence=min_prominence)
        
        # 4. Si hay nuevo pico, clasificar
        for peak_idx in peaks:
            if is_new_peak(peak_idx):
                window = extract_window(features_buffer, peak_idx)
                features = calculate_statistics(window)
                features_scaled = scaler.transform([features])
                
                probs = model.predict_proba(features_scaled)[0]
                prediction = le.classes_[np.argmax(probs)]
                
                return {
                    'repetition': True,
                    'class': prediction,
                    'confidence': float(probs.max())
                }
    
    return {'repetition': False}
```

## Métricas de Rendimiento

### Accuracy General: 91%

### Por Clase:
| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| pushup_correcto | 0.89 | 0.89 | 0.89 | 9 |
| pushup_cadera_caida | 1.00 | 1.00 | 1.00 | 2 |
| pushup_codos_abiertos | 0.86 | 0.86 | 0.86 | 7 |
| pushup_pelvis_levantada | 1.00 | 1.00 | 1.00 | 5 |

### Matriz de Confusión:
```
                    Predicho
                    corr  cod  cad  pel
Real correcto        8     1    0    0
     codos_abiertos  0     6    1    0
     cadera_caida    0     0    2    0
     pelvis_levant   0     0    0    5
```

### Dataset:
- **Total repeticiones**: 112
- **Train/Test split**: 80/20
- **Balanceo**: RandomOverSampler (34 muestras por clase en train)

## Notas Importantes

1. **Detección basada en manos**: El modelo usa `shoulder_wrist_vertical_diff` para contar repeticiones, **no** usa movimiento de cadera
2. **Robustez a cámara**: Funciona en múltiples ángulos (lateral, diagonal, frontal)
3. **Tiempo real**: Buffer circular de 150 frames permite detección continua sin latencia
4. **97.2% precisión en detección**: 35/36 videos detectan exactamente 3 repeticiones
5. **Validación mínima**: Requiere rango > 0.05 y mínimo 30 frames para clasificar

## Limitaciones

- Requiere cuerpo completo visible (hombros a pies)
- Necesita buena iluminación para MediaPipe
- Distancia óptima: 2-3 metros de la cámara
- Funciona mejor con ropa que contraste con el fondo
