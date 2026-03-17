"""
Chatbot API Fix Script
This script helps diagnose and fix chatbot API connection issues
"""

import os
from dotenv import load_dotenv

def check_chatbot_status():
    """Check the current status of the chatbot API connection"""
    print("🔍 Checking Chatbot API Status...")
    print("=" * 50)
    
    # Load environment
    load_dotenv()
    
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    print(f"🔑 API Key Status: {'✅ Present' if api_key else '❌ Missing'}")
    
    if api_key:
        print(f"🔑 API Key Length: {len(api_key)} characters")
        print(f"🔑 API Key Format: {'✅ Valid format' if api_key.startswith('AIza') else '❌ Invalid format'}")
        if api_key == "your_gemini_api_key_here":
            print("⚠️  API Key is still the placeholder - needs to be updated")
        elif len(api_key) < 10:
            print("⚠️  API Key seems too short")
    
    # Check packages
    try:
        import google.genai as genai
        print("📦 Google GenAI Package: ✅ Installed (new version)")
    except ImportError:
        try:
            import google.generativeai as genai
            print("📦 Google GenerativeAI Package: ⚠️  Installed (deprecated version)")
            print("💡 Consider upgrading: pip install google-genai")
        except ImportError:
            print("📦 Google GenerativeAI Package: ❌ Not installed")
            print("💡 Install with: pip install google-genai")
    
    # Test connection if possible
    if api_key and api_key.startswith('AIza') and len(api_key) > 10:
        try:
            import google.genai as genai
            genai.configure(api_key=api_key)
            
            # Try to list models
            try:
                models = genai.list_models()
                available_models = [model.name for model in models if 'generateContent' in model.supported_generation_methods]
                print(f"🤖 Available Models: {len(available_models)} found")
                for model in available_models[:3]:  # Show first 3
                    print(f"   - {model}")
            except Exception as e:
                error_msg = str(e).lower()
                if "leaked" in error_msg:
                    print("❌ API Key Error: Key has been reported as leaked")
                    print("💡 You need to generate a new API key from Google AI Studio")
                elif "403" in error_msg:
                    print("❌ API Key Error: Authentication failed")
                    print("💡 Check if the API key is correct")
                else:
                    print(f"❌ API Error: {e}")
        except Exception as e:
            print(f"❌ Connection Test Failed: {e}")
    
    print("\n" + "=" * 50)
    print("🔧 Fix Recommendations:")
    
    if not api_key or api_key == "your_gemini_api_key_here":
        print("1. Get a new API key from: https://aistudio.google.com/app/apikey")
        print("2. Update the .env file with: GEMINI_API_KEY=your_new_key")
    
    try:
        import google.genai
    except ImportError:
        try:
            import google.generativeai
        except ImportError:
            print("3. Install the required package: pip install google-genai")
    
    print("4. Restart the Django server after making changes")
    print("5. Test the chatbot in the application")

def update_env_file(new_api_key):
    """Update the .env file with a new API key"""
    env_file = '.env'
    
    # Read current content
    try:
        with open(env_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []
    
    # Update or add API key
    updated = False
    for i, line in enumerate(lines):
        if line.startswith('GEMINI_API_KEY='):
            lines[i] = f'GEMINI_API_KEY={new_api_key}\n'
            updated = True
            break
    
    if not updated:
        lines.append(f'GEMINI_API_KEY={new_api_key}\n')
    
    # Write back
    with open(env_file, 'w') as f:
        f.writelines(lines)
    
    print(f"✅ Updated {env_file} with new API key")

if __name__ == "__main__":
    check_chatbot_status()
    
    # Interactive mode to update API key
    print("\n" + "=" * 50)
    print("🔧 Interactive Fix Mode:")
    choice = input("Do you want to update your API key? (y/n): ").lower()
    
    if choice == 'y':
        new_key = input("Enter your new Gemini API key: ").strip()
        if new_key and new_key.startswith('AIza') and len(new_key) > 10:
            update_env_file(new_key)
            print("✅ API key updated successfully!")
            print("🔄 Please restart the Django server to apply changes")
        else:
            print("❌ Invalid API key format. Please check the key and try again.")
    else:
        print("👋 Skipping API key update.")
