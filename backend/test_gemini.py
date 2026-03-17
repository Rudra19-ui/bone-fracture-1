import os
import PIL.Image as PILImage
try:
    import google.genai as genai
    HAS_NEW_GENAI = True
    print("Using new google.genai package")
except ImportError:
    try:
        import google.generativeai as genai
        HAS_NEW_GENAI = False
        print("Using deprecated google.generativeai package")
    except ImportError:
        HAS_NEW_GENAI = False
        print("google.genai package not available")
        exit(1)

from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: GEMINI_API_KEY not found in .env")
    exit(1)

genai.configure(api_key=api_key)

# Test with different model names
model_names = ["gemini-1.5-flash", "gemini-1.0-flash", "gemini-pro"]
model = None

for model_name in model_names:
    try:
        model = genai.GenerativeModel(model_name)
        # Test the model with a simple prompt
        response = model.generate_content("Say 'Gemini is active'")
        if response.text:
            print(f"✅ Model {model_name}: {response.text.strip()}")
            break
    except Exception as e:
        print(f"❌ Model {model_name} failed: {e}")

if not model:
    print("❌ All models failed")
else:
    print(f"✅ Using model: {model_name}")

# If we have an image path from the user's latest upload, we could test it.
# Since I don't have the path, let's look at the media directory to see if I can find it.
