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
    """Handler para pushup - solo clasifica frames enviados por el cliente"""
    model_data = models["pushup"]
    model = model_data["model"]
    scaler = model_data["scaler"]
    encoder = model_data["encoder"]
    sensitivity = model_data["sensitivity"]
    base_features = BASE_FEATURES["pushup"]
    
    try:
        while True:
            t0 = time.perf_counter()
            data = await websocket.receive_json()
            
            # El cliente envía un array de frames cuando detecta una repetición
            if "frames" not in data:
                await websocket.send_json({"error": "Falta campo 'frames' (array de frames)"})
                continue
            
            frames = data["frames"]
            if not isinstance(frames, list) or len(frames) == 0:
                await websocket.send_json({"error": "'frames' debe ser un array no vacío"})
                continue
            
            # Validar que cada frame tenga las features base correctas
            for i, frame in enumerate(frames):
                missing = [f for f in base_features if f not in frame]
                if missing:
                    await websocket.send_json({
                        "error": f"Frame {i} falta features: {missing}"
                    })
                    break
            else:
                # Calcular estadísticas sobre todos los frames
                features = calculate_statistics(frames, base_features)
                features_scaled = scaler.transform(features.reshape(1, -1))
                
                # Predicción con sensibilidad aplicada
                proba = model.predict_proba(features_scaled)[0]
                sens_multipliers = np.array([sensitivity.get(cls, 1.0) for cls in encoder.classes_])
                adjusted_proba = proba * sens_multipliers
                adjusted_proba = adjusted_proba / adjusted_proba.sum()
                
                pred_idx = int(np.argmax(adjusted_proba))
                pred_label = encoder.classes_[pred_idx]
                confidence = float(adjusted_proba[pred_idx])
                
                t1 = time.perf_counter()
                
                await websocket.send_json({
                    "type": "classification",
                    "prediction": pred_label,
                    "confidence": confidence,
                    "probabilities": {cls: float(prob) for cls, prob in zip(encoder.classes_, adjusted_proba)},
                    "frames_received": len(frames),
                    "timing": {"total_ms": round((t1-t0)*1000, 2)}
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
    """Handler para squat - solo clasifica frames enviados por el cliente"""
    model_data = models["squat"]
    model = model_data["model"]
    scaler = model_data["scaler"]
    encoder = model_data["encoder"]
    sensitivity = model_data["sensitivity"]
    base_features = BASE_FEATURES["squat"]
    
    try:
        while True:
            t0 = time.perf_counter()
            data = await websocket.receive_json()
            
            # El cliente envía un array de frames cuando detecta una repetición
            if "frames" not in data:
                await websocket.send_json({"error": "Falta campo 'frames' (array de frames)"})
                continue
            
            frames = data["frames"]
            if not isinstance(frames, list) or len(frames) == 0:
                await websocket.send_json({"error": "'frames' debe ser un array no vacío"})
                continue
            
            # Validar que cada frame tenga las features base correctas
            for i, frame in enumerate(frames):
                missing = [f for f in base_features if f not in frame]
                if missing:
                    await websocket.send_json({
                        "error": f"Frame {i} falta features: {missing}"
                    })
                    break
            else:
                # Calcular estadísticas sobre todos los frames
                features = calculate_statistics(frames, base_features)
                features_scaled = scaler.transform(features.reshape(1, -1))
                
                # Predicción con sensibilidad aplicada
                proba = model.predict_proba(features_scaled)[0]
                sens_multipliers = np.array([sensitivity.get(cls, 1.0) for cls in encoder.classes_])
                adjusted_proba = proba * sens_multipliers
                adjusted_proba = adjusted_proba / adjusted_proba.sum()
                
                pred_idx = int(np.argmax(adjusted_proba))
                pred_label = encoder.classes_[pred_idx]
                confidence = float(adjusted_proba[pred_idx])
                
                t1 = time.perf_counter()
                
                await websocket.send_json({
                    "type": "classification",
                    "prediction": pred_label,
                    "confidence": confidence,
                    "probabilities": {cls: float(prob) for cls, prob in zip(encoder.classes_, adjusted_proba)},
                    "frames_received": len(frames),
                    "timing": {"total_ms": round((t1-t0)*1000, 2)}
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
    """Handler para plank - solo clasifica frames enviados por el cliente"""
    model_data = models["plank"]
    model = model_data["model"]
    scaler = model_data["scaler"]
    encoder = model_data["encoder"]
    sensitivity = model_data["sensitivity"]
    base_features = BASE_FEATURES["plank"]
    
    try:
        while True:
            t0 = time.perf_counter()
            data = await websocket.receive_json()
            
            # El cliente envía un array de frames
            if "frames" not in data:
                await websocket.send_json({"error": "Falta campo 'frames' (array de frames)"})
                continue
            
            frames = data["frames"]
            if not isinstance(frames, list) or len(frames) == 0:
                await websocket.send_json({"error": "'frames' debe ser un array no vacío"})
                continue
            
            # Validar que cada frame tenga las features base correctas
            for i, frame in enumerate(frames):
                missing = [f for f in base_features if f not in frame]
                if missing:
                    await websocket.send_json({
                        "error": f"Frame {i} falta features: {missing}"
                    })
                    break
            else:
                # Calcular estadísticas sobre todos los frames
                features = calculate_statistics(frames, base_features)
                features_scaled = scaler.transform(features.reshape(1, -1))
                
                # Predicción con sensibilidad aplicada
                proba = model.predict_proba(features_scaled)[0]
                sens_multipliers = np.array([sensitivity.get(cls, 1.0) for cls in encoder.classes_])
                adjusted_proba = proba * sens_multipliers
                adjusted_proba = adjusted_proba / adjusted_proba.sum()
                
                pred_idx = int(np.argmax(adjusted_proba))
                pred_label = encoder.classes_[pred_idx]
                confidence = float(adjusted_proba[pred_idx])
                
                t1 = time.perf_counter()
                
                await websocket.send_json({
                    "type": "classification",
                    "prediction": pred_label,
                    "confidence": confidence,
                    "probabilities": {cls: float(prob) for cls, prob in zip(encoder.classes_, adjusted_proba)},
                    "frames_received": len(frames),
                    "timing": {"total_ms": round((t1-t0)*1000, 2)}
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
    WebSocket para flexiones - El cliente envía arrays de frames:
    {
        "frames": [
            {
                "body_angle": float,
                "hip_shoulder_vertical_diff": float,
                "hip_ankle_vertical_diff": float,
                "shoulder_elbow_angle": float,
                "wrist_shoulder_hip_angle": float,
                "shoulder_wrist_vertical_diff": float
            },
            ...
        ]
    }
    El servidor calcula estadísticas (mean, std, min, max, range) y clasifica.
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
    WebSocket para sentadillas - El cliente envía arrays de frames:
    {
        "frames": [
            {
                "avg_knee_angle": float,
                "avg_hip_angle": float,
                "knee_distance": float,
                "hip_shoulder_distance": float
            },
            ...
        ]
    }
    El servidor calcula estadísticas (mean, std, min, max, range) y clasifica.
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
    WebSocket para planchas - El cliente envía arrays de frames:
    {
        "frames": [
            {
                "body_angle": float,
                "hip_shoulder_vertical_diff": float,
                "hip_ankle_vertical_diff": float,
                "shoulder_elbow_angle": float,
                "wrist_shoulder_hip_angle": float
            },
            ...
        ]
    }
    El servidor calcula estadísticas (mean, std, min, max, range) y clasifica.
    """
    await websocket.accept()
    if "plank" not in models:
        await websocket.send_json({"error": "Modelo plank no disponible"})
        await websocket.close()
        return
    await predict_with_plank(websocket)
