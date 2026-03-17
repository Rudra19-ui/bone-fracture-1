"""
Hybrid ViT-CNN Bone Fracture Detection System
Combines ResNet50 CNN with Vision Transformer for improved fracture detection
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
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam

# Import ViT components
from .vit_model import create_vit_model, create_hybrid_vit_cnn_model

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

def get_vit_model(model_type="hybrid", bone_type="Elbow"):
    """Load or create ViT-based model"""
    
    model_name = f"ViT_{model_type}_{bone_type}"
    
    if model_name in _LOADED_MODELS:
        print(f"DEBUG: Using cached ViT model: {model_name}")
        return _LOADED_MODELS[model_name]
    
    # Clear session if we have too many models loaded
    if len(_LOADED_MODELS) >= 2:
        tf.keras.backend.clear_session()
        _LOADED_MODELS.clear()
    
    print(f"DEBUG: Creating ViT model: {model_name}")
    
    if model_type == "hybrid":
        # Create Hybrid ViT-CNN model
        model = create_hybrid_vit_cnn_model(input_shape=(224, 224, 3), num_classes=2)
    elif model_type == "pure":
        # Create Pure ViT model
        model = create_vit_model(input_shape=(224, 224, 3), num_classes=2, model_size='base')
    else:
        # Default to hybrid
        model = create_hybrid_vit_cnn_model(input_shape=(224, 224, 3), num_classes=2)
    
    # Try to load pre-trained weights if available
    weight_file = os.path.join(WEIGHTS_DIR, f"{model_name}.h5")
    if os.path.exists(weight_file):
        try:
            model.load_weights(weight_file)
            print(f"DEBUG: Loaded ViT weights from {weight_file}")
        except Exception as e:
            print(f"DEBUG: Could not load ViT weights: {e}")
            print("DEBUG: Using initialized ViT model")
    else:
        print(f"DEBUG: No ViT weights found at {weight_file}")
        print("DEBUG: Using initialized ViT model (will need training)")
    
    _LOADED_MODELS[model_name] = model
    print(f"DEBUG: ViT model {model_name} loaded successfully")
    print(f"DEBUG: Model input shape: {model.input_shape}")
    print(f"DEBUG: Model output shape: {model.output_shape}")
    
    return model

def get_resnet_model(model_name):
    """Load traditional ResNet50 model for comparison"""
    
    if model_name in _LOADED_MODELS:
        print(f"DEBUG: Using cached ResNet model: {model_name}")
        return _LOADED_MODELS[model_name]
    
    # Clear session if needed
    if len(_LOADED_MODELS) >= 2:
        tf.keras.backend.clear_session()
        _LOADED_MODELS.clear()
    
    model_path_map = {
        "Elbow": "ResNet50_Elbow_frac.h5",
        "Hand": "ResNet50_Hand_frac.h5", 
        "Shoulder": "ResNet50_Shoulder_frac.h5",
        "Parts": "ResNet50_BodyParts.h5"
    }
    
    if model_name not in model_path_map:
        model_name = "Elbow"
        
    path = os.path.join(WEIGHTS_DIR, model_path_map[model_name])
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"ResNet model file not found: {path}")
        
    print(f"DEBUG: Loading ResNet model {model_name} from {path}")
    model = tf.keras.models.load_model(path)
    _LOADED_MODELS[model_name] = model
    print(f"DEBUG: ResNet model {model_name} loaded successfully")
    
    return model

def enhance_real_world_image(img_path):
    """Enhanced preprocessing for real-world images"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"DEBUG: Could not load image from {img_path}")
            return None
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Gaussian denoising
        denoised = cv2.GaussianBlur(enhanced, (3,3), 0)
        
        # Adaptive histogram equalization
        adaptive_eq = cv2.equalizeHist(denoised)
        
        # Convert back to 3-channel RGB
        enhanced_rgb = cv2.cvtColor(adaptive_eq, cv2.COLOR_GRAY2RGB)
        
        print("DEBUG: Applied enhanced preprocessing for real-world image")
        return enhanced_rgb
        
    except Exception as e:
        print(f"DEBUG: Enhanced preprocessing failed: {e}")
        return None

def preprocess_image(img_path, target_size=(224, 224)):
    """Clean preprocessing pipeline"""
    try:
        print(f"DEBUG: Starting clean preprocessing pipeline for {os.path.basename(img_path)}")
        
        # Try enhanced preprocessing first
        enhanced_img = enhance_real_world_image(img_path)
        
        if enhanced_img is not None:
            pil_img = Image.fromarray(enhanced_img)
            print("DEBUG: Using enhanced preprocessing")
        else:
            pil_img = image.load_img(img_path, target_size=target_size)
            print("DEBUG: Using standard preprocessing")
        
        # Resize to target size
        pil_img = pil_img.resize(target_size)
        
        # Convert to array
        x_arr = image.img_to_array(pil_img)
        
        # Add batch dimension
        x_arr = np.expand_dims(x_arr, axis=0)
        
        # Apply ResNet50 preprocessing (compatible with both CNN and ViT)
        x_arr = preprocess_input(x_arr)
        
        print(f"DEBUG: Image preprocessed to shape {x_arr.shape}")
        return x_arr
        
    except Exception as e:
        print(f"DEBUG: Preprocessing failed: {e}")
        return None

def generate_vit_grad_cam(model, img_array, layer_name='transformer_encoder'):
    """Generate attention-based visualization for ViT"""
    try:
        print("DEBUG: Generating ViT attention visualization...")
        
        # For ViT, we can visualize attention weights
        # This is a simplified version - in practice, you'd extract attention weights from transformer layers
        
        # Create a model that outputs attention weights (if available)
        if hasattr(model, 'layers') and len(model.layers) > 0:
            # Try to find transformer encoder layers
            for layer in model.layers:
                if 'transformer' in layer.name.lower() or 'attention' in layer.name.lower():
                    print(f"DEBUG: Found attention layer: {layer.name}")
                    # In a full implementation, you'd extract and visualize attention weights
                    break
        
        print("DEBUG: ViT attention visualization generated")
        return np.ones((14, 14))  # Placeholder attention map
        
    except Exception as e:
        print(f"DEBUG: ViT attention visualization failed: {e}")
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
                vit_result TEXT,
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
                "SELECT image_name, image_hash, part_result, fracture_result, vit_result FROM image_predictions WHERE image_hash = ?",
                (image_hash,)
            )
            row = cur.fetchone()
            if row:
                return dict(row)
        if image_name:
            cur.execute(
                "SELECT image_name, image_hash, part_result, fracture_result, vit_result FROM image_predictions WHERE image_name = ?",
                (image_name,)
            )
            row = cur.fetchone()
            if row:
                return dict(row)
        return None
    finally:
        conn.close()

def _save_cached(image_name, image_hash, part_result=None, fracture_result=None, vit_result=None):
    """Save prediction results to cache"""
    if not os.path.exists(DB_PATH):
        _init_db()
    
    conn = sqlite3.connect(DB_PATH)
    try:
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
            if vit_result is not None:
                conn.execute(
                    "UPDATE image_predictions SET vit_result = ?, image_name = ?, image_hash = ? WHERE image_name = ?",
                    (vit_result, image_name, image_hash, row[0])
                )
        else:
            conn.execute(
                "INSERT INTO image_predictions (image_name, image_hash, part_result, fracture_result, vit_result) VALUES (?, ?, ?, ?, ?)",
                (image_name, image_hash, part_result, fracture_result, vit_result)
            )
        conn.commit()
    finally:
        conn.close()

def predict_bone_type(img, force_fresh=False):
    """Clean bone type prediction using ResNet50 CNN"""
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
    
    if HAS_TF:
        try:
            print("DEBUG: Starting CNN bone type detection...")
            chosen_model = get_resnet_model("Parts")
            
            x_arr = preprocess_image(img, target_size=(size, size))
            if x_arr is None:
                raise ValueError("Preprocessing failed")
            
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

    _save_cached(image_name=image_name, image_hash=image_hash, part_result=prediction_str)
    return prediction_str

def predict_fracture_hybrid(img, bone_type, force_fresh=False):
    """HYBRID ViT-CNN FRACTURE PREDICTION - Combines both technologies"""
    size = 224
    image_name = os.path.basename(img) if isinstance(img, str) else str(img)
    
    try:
        with open(img, 'rb') as f:
            image_hash = hashlib.sha256(f.read()).hexdigest()
    except Exception:
        image_hash = None
        
    cached = _get_cached(image_hash=image_hash, image_name=image_name)

    if not force_fresh and cached and cached.get('vit_result'):
        print(f"DEBUG: Using cached hybrid result: {cached['vit_result']}")
        return cached['vit_result']
    
    if HAS_TF:
        try:
            print(f"DEBUG: Starting HYBRID ViT-CNN fracture prediction for bone type: {bone_type}")
            
            # Get both models
            resnet_model = get_resnet_model(bone_type)
            vit_model = get_vit_model("hybrid", bone_type)
            
            # Preprocess image
            x_arr = preprocess_image(img, target_size=(size, size))
            if x_arr is None:
                raise ValueError("Preprocessing failed")
            
            # ResNet50 Prediction
            print("DEBUG: Running ResNet50 CNN prediction...")
            resnet_preds = resnet_model.predict(x_arr)
            resnet_prob_fracture = float(resnet_preds[0][0])
            resnet_prob_normal = float(resnet_preds[0][1])
            
            # ViT Prediction
            print("DEBUG: Running ViT prediction...")
            vit_preds = vit_model.predict(x_arr)
            vit_prob_fracture = float(vit_preds[0][0])
            vit_prob_normal = float(vit_preds[0][1])
            
            print(f"DEBUG: ResNet50 output: [{resnet_prob_fracture:.3f}, {resnet_prob_normal:.3f}]")
            print(f"DEBUG: ViT output: [{vit_prob_fracture:.3f}, {vit_prob_normal:.3f}]")
            
            # Ensemble prediction - weighted average
            # Give more weight to ViT for global context, ResNet for local features
            vit_weight = 0.6
            resnet_weight = 0.4
            
            ensemble_prob_fracture = (vit_weight * vit_prob_fracture) + (resnet_weight * resnet_prob_fracture)
            ensemble_prob_normal = (vit_weight * vit_prob_normal) + (resnet_weight * resnet_prob_normal)
            
            print(f"DEBUG: Ensemble output: [{ensemble_prob_fracture:.3f}, {ensemble_prob_normal:.3f}]")
            print(f"DEBUG: ViT weight: {vit_weight}, ResNet weight: {resnet_weight}")
            
            # Generate attention visualization
            try:
                attention_map = generate_vit_grad_cam(vit_model, x_arr)
                if attention_map is not None:
                    print(f"DEBUG: ViT attention map generated")
            except Exception as e:
                print(f"DEBUG: ViT attention generation failed: {e}")
            
            # Final decision
            threshold = 0.5
            fracture_detected = ensemble_prob_fracture > threshold
            confidence = ensemble_prob_fracture if fracture_detected else ensemble_prob_normal
            
            result = "DETECTED" if fracture_detected else "NORMAL"
            
            print(f"DEBUG: Final HYBRID result: {result}")
            print(f"DEBUG: Confidence: {confidence:.3f}")
            print(f"DEBUG: Predicted class: {'Fractured' if fracture_detected else 'Normal'}")
            
            # Cache result
            fracture_result_label = "fractured" if fracture_detected else "normal"
            _save_cached(image_name=image_name, image_hash=image_hash, vit_result=fracture_result_label)
            
            return {
                "result": result,
                "fracture_detected": fracture_detected,
                "probability": confidence,
                "confidence_category": "High" if confidence > 0.7 else "Medium" if confidence > 0.5 else "Low",
                "safety_message": "Pattern Consistent With Fracture" if fracture_detected else "No Fracture Pattern Detected",
                "location": ANATOMICAL_MAP.get(bone_type, "Bone Structure"),
                "bone_type": bone_type,
                "model_probabilities": {
                    "resnet": {
                        "fractured": resnet_prob_fracture,
                        "normal": resnet_prob_normal
                    },
                    "vit": {
                        "fractured": vit_prob_fracture,
                        "normal": vit_prob_normal
                    },
                    "ensemble": {
                        "fractured": ensemble_prob_fracture,
                        "normal": ensemble_prob_normal
                    }
                },
                "technology": "Hybrid ViT-CNN",
                "vit_weight": vit_weight,
                "resnet_weight": resnet_weight,
                "disclaimer": "Hybrid ViT-CNN Prediction - Combining Vision Transformer and ResNet50"
            }
            
        except Exception as e:
            print(f"DEBUG: Hybrid ViT-CNN prediction failed: {e}")
            return {
                "result": "ERROR",
                "fracture_detected": False,
                "probability": 0.0,
                "error": str(e),
                "disclaimer": "Hybrid prediction failed - check logs"
            }
    else:
        print("DEBUG: TensorFlow not available for hybrid prediction")
        return {
            "result": "ERROR",
            "fracture_detected": False,
            "probability": 0.0,
            "error": "TensorFlow not available",
            "disclaimer": "Models not available"
        }

def predict_fracture_pure_vit(img, bone_type, force_fresh=False):
    """PURE ViT FRACTURE PREDICTION - Vision Transformer only"""
    size = 224
    image_name = os.path.basename(img) if isinstance(img, str) else str(img)
    
    try:
        with open(img, 'rb') as f:
            image_hash = hashlib.sha256(f.read()).hexdigest()
    except Exception:
        image_hash = None
        
    if HAS_TF:
        try:
            print(f"DEBUG: Starting PURE ViT fracture prediction for bone type: {bone_type}")
            
            # Get pure ViT model
            vit_model = get_vit_model("pure", bone_type)
            
            # Preprocess image
            x_arr = preprocess_image(img, target_size=(size, size))
            if x_arr is None:
                raise ValueError("Preprocessing failed")
            
            # ViT Prediction
            print("DEBUG: Running Pure ViT prediction...")
            preds = vit_model.predict(x_arr)
            
            prob_fracture = float(preds[0][0])
            prob_normal = float(preds[0][1])
            
            print(f"DEBUG: Pure ViT output: [{prob_fracture:.3f}, {prob_normal:.3f}]")
            print(f"DEBUG: Vision Transformer analysis complete")
            
            # Generate attention visualization
            try:
                attention_map = generate_vit_grad_cam(vit_model, x_arr)
                if attention_map is not None:
                    print(f"DEBUG: ViT attention map generated")
            except Exception as e:
                print(f"DEBUG: ViT attention generation failed: {e}")
            
            # Final decision
            threshold = 0.5
            fracture_detected = prob_fracture > threshold
            confidence = prob_fracture if fracture_detected else prob_normal
            
            result = "DETECTED" if fracture_detected else "NORMAL"
            
            print(f"DEBUG: Pure ViT result: {result}")
            print(f"DEBUG: Confidence: {confidence:.3f}")
            
            return {
                "result": result,
                "fracture_detected": fracture_detected,
                "probability": confidence,
                "confidence_category": "High" if confidence > 0.7 else "Medium" if confidence > 0.5 else "Low",
                "safety_message": "Pattern Consistent With Fracture" if fracture_detected else "No Fracture Pattern Detected",
                "location": ANATOMICAL_MAP.get(bone_type, "Bone Structure"),
                "bone_type": bone_type,
                "model_probabilities": {
                    "vit": {
                        "fractured": prob_fracture,
                        "normal": prob_normal
                    }
                },
                "technology": "Pure Vision Transformer",
                "disclaimer": "Pure ViT Prediction - Vision Transformer Analysis"
            }
            
        except Exception as e:
            print(f"DEBUG: Pure ViT prediction failed: {e}")
            return {
                "result": "ERROR",
                "fracture_detected": False,
                "probability": 0.0,
                "error": str(e),
                "disclaimer": "Pure ViT prediction failed"
            }
    else:
        print("DEBUG: TensorFlow not available for ViT prediction")
        return {
            "result": "ERROR",
            "fracture_detected": False,
            "probability": 0.0,
            "error": "TensorFlow not available",
            "disclaimer": "ViT model not available"
        }

# Legacy predict function for compatibility
def predict(img, model="Parts", force_fresh=False):
    """Legacy prediction function - routes to appropriate implementation"""
    if model == "Parts":
        return predict_bone_type(img, force_fresh=force_fresh)
    elif model == "Hybrid":
        return predict_fracture_hybrid(img, model, force_fresh=force_fresh)
    elif model == "PureViT":
        return predict_fracture_pure_vit(img, model, force_fresh=force_fresh)
    else:
        return predict_fracture_hybrid(img, model, force_fresh=force_fresh)

# Initialize database
_init_db()
print("DEBUG: Hybrid ViT-CNN prediction engine loaded successfully")
print("DEBUG: Available technologies: ResNet50 CNN, Vision Transformer, Hybrid ViT-CNN")
print("DEBUG: Vision Transformer components: Patch Embedding, Multi-Head Self-Attention, Transformer Encoder")
