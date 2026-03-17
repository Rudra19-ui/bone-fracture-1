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
        
        # Test with a simple model
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content("Hello, this is a test")
            print(f"✅ Model test successful: {response.text}")
        except Exception as e:
            print(f"❌ Model test failed: {e}")
    else:
        print("❌ No API key found")
        
except ImportError as e:
    print(f"❌ Failed to import google.generativeai: {e}")
