import os
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
if HAS_GEMINI:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    
    print("Available models:")
    try:
        models = genai.list_models()
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                print(f"- {model.name}")
    except Exception as e:
        print(f"Error listing models: {e}")
