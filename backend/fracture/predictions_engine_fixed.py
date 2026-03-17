"""
Bone Fracture Detection - Fixed Prediction Engine
CLEAN PREDICTION PIPELINE - CNN PRIMARY, NO FALLBACK OVERRIDES
"""

import os
import sys
import hashlib
import sqlite3
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

# Global variables
HAS_TF = True
_LOADED_MODELS = {}
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'weights')
DB_PATH = os.path.join(os.path.dirname(__file__), 'image_predictions.db')

# Categories and mappings
categories_parts = ['Elbow', 'Hand', 'Shoulder', 'Wrist', 'Ankle']
categories_fracture = ['fractured', 'normal']  # index 0 = fractured, index 1 = normal

ANATOMICAL_MAP = {
    "Elbow": "Humerus / Ulna / Radius",
    "Hand": "Metacarpals / Phalanges", 
    "Shoulder": "Clavicle / Humerus Head",
    "Wrist": "Distal Radius / Ulna",
    "Ankle": "Tibia / Fibula"
}

def get_model(model_name):
    """Load model with proper session management"""
    global _LOADED_MODELS
    
    if not HAS_TF:
        raise ImportError("TensorFlow is not available for ResNet inference.")
    
    # Clear session if we have too many models loaded
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
        
    print(f"DEBUG: Loading model {model_name} from {path}")
    model = tf.keras.models.load_model(path)
    _LOADED_MODELS[model_name] = model
    print(f"DEBUG: Model {model_name} loaded successfully")
    return model

def enhance_real_world_image(img_path):
    """Enhanced preprocessing for real-world images (WhatsApp, mobile cameras)"""
    try:
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"DEBUG: Could not load image from {img_path}")
            return None
            
        # Convert to grayscale for processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Gaussian denoising
        denoised = cv2.GaussianBlur(enhanced, (3,3), 0)
        
        # Convert back to 3-channel RGB
        enhanced_rgb = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
        
        print("DEBUG: Applied enhanced preprocessing for real-world image")
        return enhanced_rgb
        
    except Exception as e:
        print(f"DEBUG: Enhanced preprocessing failed: {e}")
        # Fallback to standard loading
        return None

def preprocess_image(img_path, target_size=(224, 224)):
    """Clean preprocessing pipeline"""
    try:
        # Try enhanced preprocessing first
        enhanced_img = enhance_real_world_image(img_path)
        
        if enhanced_img is not None:
            # Convert PIL to maintain consistency
            pil_img = Image.fromarray(enhanced_img)
        else:
            # Standard loading
            pil_img = image.load_img(img_path, target_size=target_size)
        
        # Resize to target size
        pil_img = pil_img.resize(target_size)
        
        # Convert to array
        x_arr = image.img_to_array(pil_img)
        
        # Add batch dimension
        x_arr = np.expand_dims(x_arr, axis=0)
        
        # Apply ResNet50 preprocessing
        x_arr = preprocess_input(x_arr)
        
        print(f"DEBUG: Image preprocessed to shape {x_arr.shape}")
        return x_arr
        
    except Exception as e:
        print(f"DEBUG: Preprocessing failed: {e}")
        return None

def generate_grad_cam(model, img_array, layer_name='conv5_block3_out'):
    """Generate Grad-CAM heatmap for model interpretability"""
    try:
        # Create a model that maps the input image to the activations of the last conv layer
        grad_model = Model(
            inputs=[model.inputs], 
            outputs=[model.get_layer(layer_name).output, model.output]
        )
        
        # Compute gradient of top predicted class
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, 0]  # Fracture class (index 0)
        
        # Extract gradients
        output = conv_outputs[0]
        grads = tape.gradient(loss, conv_outputs)[0]
        
        # Global average pooling
        weights = tf.reduce_mean(grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, output), axis=-1)
        
        # Apply ReLU and normalize
        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam)
        
        print("DEBUG: Grad-CAM heatmap generated successfully")
        return cam
        
    except Exception as e:
        print(f"DEBUG: Grad-CAM generation failed: {e}")
        return None

def _init_db():
    """Initialize SQLite cache database"""
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS image_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_name TEXT,
                image_hash TEXT,
                part_result TEXT,
                fracture_result TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
    finally:
        conn.close()

def _get_cached(image_hash=None, image_name=None):
    """Get cached prediction results"""
    if not os.path.exists(DB_PATH):
        _init_db()
    
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        if image_hash:
            cur.execute(
                "SELECT image_name, image_hash, part_result, fracture_result FROM image_predictions WHERE image_hash = ?",
                (image_hash,)
            )
            row = cur.fetchone()
            if row:
                return dict(row)
        if image_name:
            cur.execute(
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
    """Save prediction results to cache"""
    if not os.path.exists(DB_PATH):
        _init_db()
    
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
            # Update existing record
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

def predict_bone_type(img, force_fresh=False):
    """Clean bone type prediction using only CNN model"""
    size = 224
    image_name = os.path.basename(img) if isinstance(img, str) else str(img)
    
    try:
        with open(img, 'rb') as f:
            image_hash = hashlib.sha256(f.read()).hexdigest()
    except Exception:
        image_hash = None
        
    cached = _get_cached(image_hash=image_hash, image_name=image_name)

    if not force_fresh and cached and cached.get('part_result'):
        print(f"DEBUG: Using cached bone type result: {cached['part_result']}")
        return cached['part_result']
    
    prediction_str = "Unknown"
    
    # Use CNN model only - NO FALLBACKS
    if HAS_TF:
        try:
            print("DEBUG: Starting CNN bone type detection...")
            chosen_model = get_model("Parts")
            
            # Clean preprocessing
            x_arr = preprocess_image(img, target_size=(size, size))
            if x_arr is None:
                raise ValueError("Preprocessing failed")
            
            # Model prediction
            preds = chosen_model.predict(x_arr)
            prediction_idx = np.argmax(preds, axis=1).item()
            prediction_str = categories_parts[prediction_idx] if prediction_idx < len(categories_parts) else "Unknown"
            
            print(f"DEBUG: CNN bone type prediction: {prediction_str}")
            print(f"DEBUG: Bone type probabilities: {preds}")
            
        except Exception as e:
            print(f"DEBUG: CNN bone type prediction failed: {e}")
            prediction_str = "Unknown"
    else:
        print("DEBUG: TensorFlow not available for bone type detection")
        prediction_str = "Unknown"

    # Cache result
    _save_cached(image_name=image_name, image_hash=image_hash, part_result=prediction_str)
    return prediction_str

def predict_fracture(img, bone_type, force_fresh=False):
    """CLEAN FRACTURE PREDICTION - CNN PRIMARY, NO FALLBACK OVERRIDES"""
    size = 224
    image_name = os.path.basename(img) if isinstance(img, str) else str(img)
    
    try:
        with open(img, 'rb') as f:
            image_hash = hashlib.sha256(f.read()).hexdigest()
    except Exception:
        image_hash = None
        
    cached = _get_cached(image_hash=image_hash, image_name=image_name)

    if not force_fresh and cached and cached.get('fracture_result'):
        print(f"DEBUG: Using cached fracture result: {cached['fracture_result']}")
        return cached['fractacture_result']
    
    # CNN MODEL PREDICTION ONLY
    if HAS_TF:
        try:
            print(f"DEBUG: Starting CNN fracture prediction for bone type: {bone_type}")
            
            # Map bone type to correct model
            model_mapping = {
                "Elbow": "Elbow",
                "Hand": "Hand", 
                "Shoulder": "Shoulder",
                "Wrist": "Hand",  # Use Hand model for Wrist
                "Ankle": "Elbow"   # Use Elbow model for Ankle
            }
            
            inference_model = model_mapping.get(bone_type, "Elbow")
            print(f"DEBUG: Using fracture model: {inference_model}")
            
            chosen_model = get_model(inference_model)
            
            # Clean preprocessing
            x_arr = preprocess_image(img, target_size=(size, size))
            if x_arr is None:
                raise ValueError("Preprocessing failed")
            
            # Model prediction
            preds = chosen_model.predict(x_arr)
            
            # Extract probabilities
            prob_fracture = float(preds[0][0])  # Index 0 = fractured
            prob_normal = float(preds[0][1])     # Index 1 = normal
            
            print(f"DEBUG: Model output: [{prob_fracture:.3f}, {prob_normal:.3f}]")
            print(f"DEBUG: Fracture probability: {prob_fracture:.3f}")
            print(f"DEBUG: Normal probability: {prob_normal:.3f}")
            
            # CLEAN DECISION RULE - CNN PRIMARY
            threshold = 0.5
            fracture_detected = prob_fracture > threshold
            
            # Generate Grad-CAM for interpretability
            try:
                grad_cam = generate_grad_cam(chosen_model, x_arr)
                if grad_cam is not None:
                    print(f"DEBUG: Grad-CAM generated for interpretability")
            except Exception as e:
                print(f"DEBUG: Grad-CAM generation failed: {e}")
            
            # Final result based ONLY on CNN prediction
            result = "DETECTED" if fracture_detected else "NORMAL"
            confidence = prob_fracture if fracture_detected else prob_normal
            
            print(f"DEBUG: Final result: {result}")
            print(f"DEBUG: Confidence: {confidence:.3f}")
            
            # Cache result
            fracture_result_label = "fractured" if fracture_detected else "normal"
            _save_cached(image_name=image_name, image_hash=image_hash, fracture_result=fracture_result_label)
            
            return {
                "result": result,
                "fracture_detected": fracture_detected,
                "probability": confidence,
                "confidence_category": "High" if confidence > 0.7 else "Medium" if confidence > 0.5 else "Low",
                "safety_message": "Pattern Consistent With Fracture" if fracture_detected else "No Fracture Pattern Detected",
                "location": ANATOMICAL_MAP.get(bone_type, "Bone Structure"),
                "bone_type": bone_type,
                "model_probabilities": {
                    "fractured": prob_fracture,
                    "normal": prob_normal
                },
                "disclaimer": "ResNet50 CNN Prediction - Clean Pipeline"
            }
            
        except Exception as e:
            print(f"DEBUG: CNN fracture prediction failed: {e}")
            return {
                "result": "ERROR",
                "fracture_detected": False,
                "probability": 0.0,
                "error": str(e),
                "disclaimer": "Prediction failed - check logs"
            }
    else:
        print("DEBUG: TensorFlow not available for fracture prediction")
        return {
            "result": "ERROR",
            "fracture_detected": False,
            "probability": 0.0,
            "error": "TensorFlow not available",
            "disclaimer": "Model not available"
        }

# Legacy predict function for compatibility
def predict(img, model="Parts", force_fresh=False):
    """Legacy prediction function - routes to clean implementation"""
    if model == "Parts":
        return predict_bone_type(img, force_fresh=force_fresh)
    else:
        return predict_fracture(img, model, force_fresh=force_fresh)

# Initialize database
_init_db()
print("DEBUG: Clean prediction engine loaded successfully")
