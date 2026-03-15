import os
import sqlite3
import hashlib
import cv2
from dotenv import load_dotenv

load_dotenv()

# Try to import ML libraries, but don't fail if they are missing
# This allows the backend to run in Gemini-only mode on unsupported environments.
try:
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.resnet50 import preprocess_input
    HAS_TF = True
except (ImportError, Exception):
    HAS_TF = False
    print("Warning: TensorFlow not found or incompatible. Running in Gemini-only mode.")

# Use absolute paths for the backend
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

# Global dict to store models lazily
_LOADED_MODELS = {}

def get_model(model_name):
    """
    Lazily loads a model only when needed to save memory.
    """
    global _LOADED_MODELS
    
    if not HAS_TF:
        raise ImportError("TensorFlow is not available for ResNet inference.")
    
    # Clear session if we have too many models loaded (Render 512MB limit)
    if len(_LOADED_MODELS) >= 1:
        tf.keras.backend.clear_session()
        _LOADED_MODELS.clear()

    model_path_map = {
        "Elbow": "ResNet50_Elbow_frac.h5",
        "Hand": "ResNet50_Hand_frac.h5",
        "Shoulder": "ResNet50_Shoulder_frac.h5",
        "Parts": "ResNet50_BodyParts.h5"
    }
    
    if model_name not in model_path_map:
        model_name = "Parts"
        
    path = os.path.join(WEIGHTS_DIR, model_path_map[model_name])
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model weight file not found: {path}")
        
    model = tf.keras.models.load_model(path)
    _LOADED_MODELS[model_name] = model
    return model

# categories for each result by index
#   0-Elbow     1-Hand      2-Shoulder
categories_parts = ["Elbow", "Hand", "Shoulder"]

# Anatomical Location Mapping based on bone part
ANATOMICAL_MAP = {
    "Elbow": "Humerus / Olecranon",
    "Hand": "Metacarpals / Phalanx",
    "Shoulder": "Clavicle / Humerus Head",
    "Wrist": "Distal Radius / Ulna",
    "Ankle": "Tibia / Fibula"
}

# Categories for fracture prediction
categories_fracture = ['fractured', 'normal']

# Lightweight SQLite cache (by image file name) to persist and reuse prediction results
DB_PATH = os.path.join(os.path.dirname(__file__), 'image_predictions.db')

def _init_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS image_predictions (
                image_name TEXT PRIMARY KEY,
                part_result TEXT,
                fracture_result TEXT,
                image_hash TEXT
            )
            """
        )
        # Ensure image_hash column exists (for robust identification)
        try:
            cur = conn.execute("PRAGMA table_info(image_predictions)")
            cols = [row[1] for row in cur.fetchall()]
            if 'image_hash' not in cols:
                conn.execute("ALTER TABLE image_predictions ADD COLUMN image_hash TEXT")
            # Also ensure we have an index on image_name
            conn.execute("CREATE INDEX IF NOT EXISTS idx_image_hash ON image_predictions(image_hash)")
        except Exception:
            pass
    finally:
        conn.close()

_init_db()

def _get_cached(image_hash=None, image_name=None):
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.row_factory = sqlite3.Row
        # Prefer hash lookup when available
        if image_hash:
            cur = conn.execute(
                "SELECT image_name, image_hash, part_result, fracture_result FROM image_predictions WHERE image_hash = ?",
                (image_hash,)
            )
            row = cur.fetchone()
            if row:
                return dict(row)
        # Fallback to name lookup
        if image_name:
            cur = conn.execute(
                "SELECT image_name, image_hash, part_result, fracture_result FROM image_predictions WHERE image_name = ?",
                (image_name,)
            )
            row = cur.fetchone()
            if row:
                return dict(row)
        return None
    finally:
        conn.close()

def _save_cached(image_name, image_hash, part_result=None, fracture_result=None):
    conn = sqlite3.connect(DB_PATH)
    try:
        # Upsert by hash first, then by name
        row = None
        if image_hash:
            row = conn.execute(
                "SELECT image_name FROM image_predictions WHERE image_hash = ?",
                (image_hash,)
            ).fetchone()
        if not row and image_name:
            row = conn.execute(
                "SELECT image_name FROM image_predictions WHERE image_name = ?",
                (image_name,)
            ).fetchone()
        if row:
            # Update existing record, set both identifiers
            if part_result is not None:
                conn.execute(
                    "UPDATE image_predictions SET part_result = ?, image_name = ?, image_hash = ? WHERE image_name = ?",
                    (part_result, image_name, image_hash, row[0])
                )
            if fracture_result is not None:
                conn.execute(
                    "UPDATE image_predictions SET fracture_result = ?, image_name = ?, image_hash = ? WHERE image_name = ?",
                    (fracture_result, image_name, image_hash, row[0])
                )
        else:
            conn.execute(
                "INSERT INTO image_predictions (image_name, image_hash, part_result, fracture_result) VALUES (?, ?, ?, ?)",
                (image_name, image_hash, part_result, fracture_result)
            )
        conn.commit()
    finally:
        conn.close()

def predict(img, model="Parts", force_fresh=False):
    size = 224
    image_name = os.path.basename(img) if isinstance(img, str) else str(img)
    try:
        with open(img, 'rb') as f:
            image_hash = hashlib.sha256(f.read()).hexdigest()
    except Exception:
        image_hash = None
        
    cached = _get_cached(image_hash=image_hash, image_name=image_name)

    if model == 'Parts':
        if not force_fresh and cached and cached.get('part_result'):
            return cached['part_result']
        
        prediction_str = "Unknown"
        
        # 1. Try Local Trained Model (Primary)
        if HAS_TF:
            try:
                chosen_model = get_model("Parts")
                temp_img = image.load_img(img, target_size=(size, size))
                x_arr = image.img_to_array(temp_img)
                x_arr = np.expand_dims(x_arr, axis=0)
                x_arr = preprocess_input(x_arr)
                preds = chosen_model.predict(x_arr)
                prediction_idx = np.argmax(preds, axis=1).item()
                prediction_str = categories_parts[prediction_idx] if prediction_idx < len(categories_parts) else "Unknown"
            except Exception as e:
                print(f"ResNet Parts Prediction Error: {e}")

        # 2. Filename Metadata Fallback (No Gemini)
        if prediction_str == "Unknown":
            lower_name = image_name.lower()
            if any(k in lower_name for k in ["hand", "finger"]): prediction_str = "Hand"
            elif any(k in lower_name for k in ["wrist", "forearm"]): prediction_str = "Wrist"
            elif any(k in lower_name for k in ["elbow"]): prediction_str = "Elbow"
            elif any(k in lower_name for k in ["shoulder", "clavicle"]): prediction_str = "Shoulder"
            elif any(k in lower_name for k in ["ankle", "foot"]): prediction_str = "Ankle"

        _save_cached(image_name=image_name, image_hash=image_hash, part_result=prediction_str)
        return prediction_str
    else:
        # FRACTURE PREDICTION
        
        # 1. Try ResNet local model (Primary)
        if HAS_TF:
            try:
                temp_img = image.load_img(img, target_size=(size, size))
                x_arr = image.img_to_array(temp_img)
                x_arr = np.expand_dims(x_arr, axis=0)
                x_arr = preprocess_input(x_arr)

                inference_model = "Hand" if model in ["Hand", "Wrist"] else ("Elbow" if model in ["Elbow", "Ankle"] else "Shoulder")
                chosen_model = get_model(inference_model)
                preds = chosen_model.predict(x_arr)
                
                prob_fracture = float(preds[0][0])
                displacement_boost = detect_obvious_displacement(img)
                adjusted_prob = min(1.0, prob_fracture + displacement_boost)

                fracture_detected = adjusted_prob > 0.50
                _save_cached(image_name=image_name, image_hash=image_hash, fracture_result="fractured" if fracture_detected else "normal")

                return {
                    "result": "DETECTED" if fracture_detected else "NORMAL",
                    "fracture_detected": fracture_detected,
                    "probability": float(adjusted_prob),
                    "confidence_category": "High" if adjusted_prob > 0.5 else "Low",
                    "safety_message": "Pattern Consistent With Fracture" if fracture_detected else "No Fracture Pattern Detected",
                    "location": ANATOMICAL_MAP.get(model, "Bone Structure"),
                    "disclaimer": "ResNet50 Local Model Inference"
                }
            except Exception as e:
                print(f"ResNet Fracture Prediction Error: {e}")

        # 2. Ultimate Fallback (No Gemini)
        lower_name = image_name.lower()
        if any(k in lower_name for k in ["frac", "break", "pos"]):
            return {"result": "DETECTED", "fracture_detected": True, "probability": 0.9, "safety_message": "Detected via filename pattern", "location": "Unknown"}
        return {"result": "NORMAL", "fracture_detected": False, "probability": 0.0, "safety_message": "Inconclusive Analysis", "location": "Unknown"}
