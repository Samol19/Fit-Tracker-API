import os
import joblib
import numpy as np
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List
from collections import deque
from scipy.signal import find_peaks, savgol_filter

app = FastAPI()

# Permitir CORS para desarrollo y producción
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración de sensibilidades para cada modelo
SENSIBILIDAD = {
    "pushup": {
        "pushup_correcto": 1.0,
        "pushup_cadera_caida": 1.0,
        "pushup_codos_abiertos": 0.9,
        "pushup_pelvis_levantada": 1.0,
    },
    "squat": {
        "squat_correcto": 1.0,
        "squat_poca_profundidad": 4.0,
        "squat_espalda_arqueada": 1.2,
        "squat_valgo_rodilla": 2.5,
    },
    "plank": {
        "plank_correcto": 1.3,
        "plank_cadera_caida": 1.0,
        "plank_codos_abiertos": 0.5,
        "plank_pelvis_levantada": 1.0,
    }
}

# Configuración de features base para cada modelo
BASE_FEATURES = {
    "pushup": [
        "body_angle",
        "hip_shoulder_vertical_diff",
        "hip_ankle_vertical_diff",
        "shoulder_elbow_angle",
        "wrist_shoulder_hip_angle",
        "shoulder_wrist_vertical_diff"
    ],
    "squat": [
        "avg_knee_angle",
        "avg_hip_angle",
        "knee_distance",
        "hip_shoulder_distance"
    ],
    "plank": [
        "body_angle",
        "hip_shoulder_vertical_diff",
        "hip_ankle_vertical_diff",
        "shoulder_elbow_angle",
        "wrist_shoulder_hip_angle"
    ]
}

# Configuración de buffers
BUFFER_CONFIG = {
    "pushup": {
        "size": 150,  # 5 segundos a 30fps
        "window_margin": 20,  # frames antes/después del pico
        "min_peak_distance": 25,
        "min_prominence": 0.03
    },
    "squat": {
        "accumulate": True  # Acumula todo el movimiento
    },
    "plank": {
        "size": 30  # 1 segundo a 30fps
    }
}

# Diccionario para almacenar los modelos cargados
models: Dict[str, dict] = {}

def load_model(exercise_type: str):
    """Carga el modelo, scaler y encoder para un tipo de ejercicio"""
    model_path = os.path.join("model", f"{exercise_type}_classifier_model.pkl")
    scaler_path = os.path.join("model", f"{exercise_type}_scaler.pkl")
    encoder_path = os.path.join("model", f"{exercise_type}_label_encoder.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    
    return {
        "model": joblib.load(model_path),
        "scaler": joblib.load(scaler_path),
        "encoder": joblib.load(encoder_path),
        "sensitivity": SENSIBILIDAD.get(exercise_type, {})
    }

# Cargar todos los modelos al iniciar
print("Cargando modelos...")
for exercise in ["pushup", "squat", "plank"]:
    try:
        models[exercise] = load_model(exercise)
        print(f"✓ Modelo {exercise} cargado")
    except Exception as e:
        print(f"✗ Error cargando modelo {exercise}: {e}")

print(f"Total modelos disponibles: {len(models)}")

@app.get("/")
def read_root():
    """Estado de la API y modelos disponibles"""
    return {
        "status": "ok",
        "models_loaded": list(models.keys()),
        "endpoints": {
            "pushup": "/ws/pushup",
            "squat": "/ws/squat",
            "plank": "/ws/plank"
        }
    }

@app.get("/models")
def list_models():
    """Información detallada de los modelos disponibles"""
    model_info = {}
    for exercise, model_data in models.items():
        base_features = BASE_FEATURES.get(exercise, [])
        model_info[exercise] = {
            "classes": model_data["encoder"].classes_.tolist(),
            "n_features_model": model_data["scaler"].n_features_in_,
            "base_features": base_features,
            "n_base_features": len(base_features),
            "sensitivity": model_data["sensitivity"],
            "buffer_config": BUFFER_CONFIG.get(exercise, {})
        }
    return model_info

def calculate_statistics(data: List[dict], base_features: List[str]) -> np.ndarray:
    """
    Calcula estadísticas (mean, std, min, max, range) para cada feature base.
    
    Args:
        data: Lista de diccionarios con valores de features base por frame
        base_features: Lista de nombres de features base
    
    Returns:
        Array con todas las estadísticas calculadas
    """
    features = []
    
    for feature_name in base_features:
        values = np.array([frame[feature_name] for frame in data])
        
        features.append(np.mean(values))
        features.append(np.std(values))
        features.append(np.min(values))
        features.append(np.max(values))
        features.append(np.max(values) - np.min(values))  # range
    
    return np.array(features)

async def predict_with_pushup(websocket: WebSocket):
    """Handler específico para pushup con detección de picos"""
    model_data = models["pushup"]
    model = model_data["model"]
    scaler = model_data["scaler"]
    encoder = model_data["encoder"]
    sensitivity = model_data["sensitivity"]
    base_features = BASE_FEATURES["pushup"]
    config = BUFFER_CONFIG["pushup"]
    
    # Buffer circular para acumular frames
    buffer = deque(maxlen=config["size"])
    signal_buffer = deque(maxlen=config["size"])
    rep_count = 0
    last_peak_idx = -config["min_peak_distance"]
    
    try:
        while True:
            t0 = time.perf_counter()
            data = await websocket.receive_json()
            t1 = time.perf_counter()
            
            # Validar que vengan las 6 features base
            if "metrics" not in data:
                await websocket.send_json({"error": "Falta campo 'metrics'"})
                continue
            
            metrics = data["metrics"]
            if len(metrics) != len(base_features):
                await websocket.send_json({
                    "error": f"Se esperan {len(base_features)} features base, recibidas: {len(metrics)}"
                })
                continue
            
            # Agregar al buffer
            buffer.append(metrics)
            signal_buffer.append(metrics["shoulder_wrist_vertical_diff"])
            
            # Detección de picos si hay suficientes frames
            if len(signal_buffer) >= 50:
                signal = np.array(signal_buffer)
                smoothed = savgol_filter(signal, min(11, len(signal) if len(signal) % 2 == 1 else len(signal)-1), 3)
                
                signal_min = np.min(smoothed)
                signal_max = np.max(smoothed)
                signal_range = signal_max - signal_min
                
                if signal_range > 0.05:  # Validación mínima
                    height_threshold = signal_min + (signal_range * 0.40)
                    prominence_min = max(signal_range * 0.10, config["min_prominence"])
                    
                    peaks, _ = find_peaks(
                        smoothed,
                        height=height_threshold,
                        distance=config["min_peak_distance"],
                        prominence=prominence_min
                    )
                    
                    # Verificar nuevos picos
                    for peak_idx in peaks:
                        if peak_idx > last_peak_idx + config["min_peak_distance"]:
                            # Extraer ventana alrededor del pico
                            start = max(0, peak_idx - config["window_margin"])
                            end = min(len(buffer), peak_idx + config["window_margin"])
                            
                            if end - start >= 30:  # Mínimo 30 frames
                                window_data = list(buffer)[start:end]
                                
                                # Calcular estadísticas
                                features = calculate_statistics(window_data, base_features)
                                features_scaled = scaler.transform(features.reshape(1, -1))
                                
                                # Predicción con sensibilidad
                                proba = model.predict_proba(features_scaled)[0]
                                sens_multipliers = np.array([sensitivity.get(cls, 1.0) for cls in encoder.classes_])
                                adjusted_proba = proba * sens_multipliers
                                adjusted_proba = adjusted_proba / adjusted_proba.sum()
                                
                                pred_idx = int(np.argmax(adjusted_proba))
                                pred_label = encoder.classes_[pred_idx]
                                confidence = float(adjusted_proba[pred_idx])
                                
                                rep_count += 1
                                last_peak_idx = peak_idx
                                
                                t2 = time.perf_counter()
                                
                                await websocket.send_json({
                                    "type": "repetition",
                                    "rep_number": rep_count,
                                    "prediction": pred_label,
                                    "confidence": confidence,
                                    "probabilities": {cls: float(prob) for cls, prob in zip(encoder.classes_, adjusted_proba)},
                                    "timing": {"total_ms": round((t2-t0)*1000, 2)}
                                })
                                continue
            
            # Respuesta de estado (sin repetición detectada)
            await websocket.send_json({
                "type": "status",
                "buffer_size": len(buffer),
                "ready": len(signal_buffer) >= 50
            })
            
    except WebSocketDisconnect:
        print(f"Cliente desconectado de pushup")
    except Exception as e:
        print(f"Error en pushup: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass

async def predict_with_squat(websocket: WebSocket):
    """Handler específico para squat con máquina de estados"""
    model_data = models["squat"]
    model = model_data["model"]
    scaler = model_data["scaler"]
    encoder = model_data["encoder"]
    sensitivity = model_data["sensitivity"]
    base_features = BASE_FEATURES["squat"]
    
    # Estado de la máquina
    estado = "ARRIBA"
    frame_buffer = []
    rep_count = 0
    
    # Umbrales
    KNEE_ANGLE_DOWN = 160
    KNEE_ANGLE_UP = 170
    KNEE_ANGLE_BOTTOM = 90
    
    try:
        while True:
            t0 = time.perf_counter()
            data = await websocket.receive_json()
            t1 = time.perf_counter()
            
            # Validar métricas
            if "metrics" not in data:
                await websocket.send_json({"error": "Falta campo 'metrics'"})
                continue
            
            metrics = data["metrics"]
            if len(metrics) != len(base_features):
                await websocket.send_json({
                    "error": f"Se esperan {len(base_features)} features base"
                })
                continue
            
            knee_angle = metrics["avg_knee_angle"]
            
            # Máquina de estados
            if estado == "ARRIBA" and knee_angle < KNEE_ANGLE_DOWN:
                estado = "BAJANDO"
                frame_buffer = [metrics]
                
            elif estado == "BAJANDO":
                frame_buffer.append(metrics)
                if knee_angle < KNEE_ANGLE_BOTTOM:
                    estado = "ABAJO"
                    
            elif estado == "ABAJO":
                frame_buffer.append(metrics)
                if knee_angle > KNEE_ANGLE_DOWN:
                    estado = "SUBIENDO"
                    
            elif estado == "SUBIENDO":
                frame_buffer.append(metrics)
                if knee_angle > KNEE_ANGLE_UP:
                    # REPETICIÓN COMPLETA
                    estado = "ARRIBA"
                    rep_count += 1
                    
                    if len(frame_buffer) >= 45:  # Mínimo 45 frames
                        # Calcular estadísticas
                        features = calculate_statistics(frame_buffer, base_features)
                        features_scaled = scaler.transform(features.reshape(1, -1))
                        
                        # Predicción con sensibilidad
                        proba = model.predict_proba(features_scaled)[0]
                        sens_multipliers = np.array([sensitivity.get(cls, 1.0) for cls in encoder.classes_])
                        adjusted_proba = proba * sens_multipliers
                        adjusted_proba = adjusted_proba / adjusted_proba.sum()
                        
                        pred_idx = int(np.argmax(adjusted_proba))
                        pred_label = encoder.classes_[pred_idx]
                        confidence = float(adjusted_proba[pred_idx])
                        
                        t2 = time.perf_counter()
                        
                        await websocket.send_json({
                            "type": "repetition",
                            "rep_number": rep_count,
                            "prediction": pred_label,
                            "confidence": confidence,
                            "probabilities": {cls: float(prob) for cls, prob in zip(encoder.classes_, adjusted_proba)},
                            "frame_count": len(frame_buffer),
                            "timing": {"total_ms": round((t2-t0)*1000, 2)}
                        })
                        continue
            
            # Estado intermedio
            await websocket.send_json({
                "type": "status",
                "state": estado,
                "knee_angle": float(knee_angle),
                "frames_accumulated": len(frame_buffer)
            })
            
    except WebSocketDisconnect:
        print(f"Cliente desconectado de squat")
    except Exception as e:
        print(f"Error en squat: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass

async def predict_with_plank(websocket: WebSocket):
    """Handler específico para plank con buffer temporal"""
    model_data = models["plank"]
    model = model_data["model"]
    scaler = model_data["scaler"]
    encoder = model_data["encoder"]
    sensitivity = model_data["sensitivity"]
    base_features = BASE_FEATURES["plank"]
    config = BUFFER_CONFIG["plank"]
    
    # Buffer para acumular métricas base de cada frame
    buffer = deque(maxlen=config["size"])
    
    try:
        while True:
            t0 = time.perf_counter()
            data = await websocket.receive_json()
            t1 = time.perf_counter()
            
            # Validar que vengan las 5 métricas base
            if "metrics" not in data:
                await websocket.send_json({"error": "Falta campo 'metrics'"})
                continue
            
            metrics = data["metrics"]
            if len(metrics) != len(base_features):
                await websocket.send_json({
                    "error": f"Se esperan {len(base_features)} features base, recibidas: {len(metrics)}"
                })
                continue
            
            # Agregar al buffer
            buffer.append(metrics)
            
            # Si el buffer está lleno, clasificar
            if len(buffer) == config["size"]:
                # Calcular estadísticas sobre el buffer completo
                features = calculate_statistics(list(buffer), base_features)
                features_scaled = scaler.transform(features.reshape(1, -1))
                
                # Predicción con sensibilidad
                proba = model.predict_proba(features_scaled)[0]
                sens_multipliers = np.array([sensitivity.get(cls, 1.0) for cls in encoder.classes_])
                adjusted_proba = proba * sens_multipliers
                adjusted_proba = adjusted_proba / adjusted_proba.sum()
                
                pred_idx = int(np.argmax(adjusted_proba))
                pred_label = encoder.classes_[pred_idx]
                confidence = float(adjusted_proba[pred_idx])
                
                t2 = time.perf_counter()
                
                await websocket.send_json({
                    "type": "classification",
                    "prediction": pred_label,
                    "confidence": confidence,
                    "probabilities": {cls: float(prob) for cls, prob in zip(encoder.classes_, adjusted_proba)},
                    "timing": {"total_ms": round((t2-t0)*1000, 2)}
                })
            else:
                # Buffer llenándose
                await websocket.send_json({
                    "type": "status",
                    "buffer_size": len(buffer),
                    "buffer_progress": len(buffer) / config["size"]
                })
            
    except WebSocketDisconnect:
        print(f"Cliente desconectado de plank")
    except Exception as e:
        print(f"Error en plank: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass

@app.websocket("/ws/pushup")
async def websocket_pushup(websocket: WebSocket):
    """
    WebSocket para flexiones - Enviar métricas base por frame:
    {
        "metrics": {
            "body_angle": float,
            "hip_shoulder_vertical_diff": float,
            "hip_ankle_vertical_diff": float,
            "shoulder_elbow_angle": float,
            "wrist_shoulder_hip_angle": float,
            "shoulder_wrist_vertical_diff": float
        }
    }
    """
    await websocket.accept()
    if "pushup" not in models:
        await websocket.send_json({"error": "Modelo pushup no disponible"})
        await websocket.close()
        return
    await predict_with_pushup(websocket)

@app.websocket("/ws/squat")
async def websocket_squat(websocket: WebSocket):
    """
    WebSocket para sentadillas - Enviar métricas base por frame:
    {
        "metrics": {
            "avg_knee_angle": float,
            "avg_hip_angle": float,
            "knee_distance": float,
            "hip_shoulder_distance": float
        }
    }
    """
    await websocket.accept()
    if "squat" not in models:
        await websocket.send_json({"error": "Modelo squat no disponible"})
        await websocket.close()
        return
    await predict_with_squat(websocket)

@app.websocket("/ws/plank")
async def websocket_plank(websocket: WebSocket):
    """
    WebSocket para planchas - Enviar métricas base por frame:
    {
        "metrics": {
            "body_angle": float,
            "hip_shoulder_vertical_diff": float,
            "hip_ankle_vertical_diff": float,
            "shoulder_elbow_angle": float,
            "wrist_shoulder_hip_angle": float
        }
    }
    """
    await websocket.accept()
    if "plank" not in models:
        await websocket.send_json({"error": "Modelo plank no disponible"})
        await websocket.close()
        return
    await predict_with_plank(websocket)
