import os
import PIL.Image as PILImage
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: GEMINI_API_KEY not found in .env")
    exit(1)

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# Test with a simple prompt
try:
    response = model.generate_content("Say 'Gemini is active'")
    print(f"Connection test: {response.text.strip()}")
except Exception as e:
    print(f"Connection test failed: {e}")

# If we have an image path from the user's latest upload, we could test it.
# Since I don't have the path, let's look at the media directory to see if I can find it.
