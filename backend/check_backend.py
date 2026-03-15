import sys
import os

# Add the current directory to path so we can import fracture
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from fracture.predictions_engine import HAS_TF, WEIGHTS_DIR, _LOADED_MODELS, predict
    print(f"HAS_TF: {HAS_TF}")
    print(f"WEIGHTS_DIR: {WEIGHTS_DIR}")
    
    # Check if models exist
    models = ["ResNet50_BodyParts.h5", "ResNet50_Elbow_frac.h5", "ResNet50_Hand_frac.h5", "ResNet50_Shoulder_frac.h5"]
    for m in models:
        path = os.path.join(WEIGHTS_DIR, m)
        print(f"Model {m} exists: {os.path.exists(path)}")
        
    # Check Gemini key
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    print(f"GEMINI_API_KEY set: {bool(api_key)}")
    if api_key:
        print(f"GEMINI_API_KEY length: {len(api_key)}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
