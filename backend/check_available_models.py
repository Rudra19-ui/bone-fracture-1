import os
from dotenv import load_dotenv

load_dotenv()

# Test new google-genai package
try:
    import google.generativeai as genai
    print("✅ Successfully imported google.generativeai")
    
    # Configure API
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        print("✅ API key configured")
        
        # List available models
        try:
            models = genai.list_models()
            print("📋 Available Models:")
            for model in models:
                if 'generateContent' in model.supported_generation_methods:
                    print(f"   ✅ {model.name}")
                else:
                    print(f"   ❌ {model.name} (no generateContent)")
        except Exception as e:
            print(f"❌ Failed to list models: {e}")
    else:
        print("❌ No API key found")
        
except ImportError as e:
    print(f"❌ Failed to import google.generativeai: {e}")
