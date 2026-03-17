import os
from dotenv import load_dotenv

def test_chatbot_connection():
    """Test the chatbot API connection"""
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
                working_models = []
                
                for model in models:
                    if 'generateContent' in model.supported_generation_methods:
                        print(f"   ✅ {model.name} - Working")
                        working_models.append(model.name)
                    else:
                        print(f"   ❌ {model.name} - No generateContent support")
                
                if working_models:
                    # Use the first working model we find
                    selected_model = working_models[0]
                    print(f"\n🎯 Using model: {selected_model}")
                    
                    # Test the selected model
                    try:
                        model = genai.GenerativeModel(selected_model)
                        response = model.generate_content("Hello, this is a test of the chatbot API")
                        print(f"✅ SUCCESS: {response.text}")
                        return True
                    except Exception as e:
                        print(f"❌ FAILED: {e}")
                        return False
                else:
                    print("❌ No working models found")
                    return False
                    
            except Exception as e:
                print(f"❌ Failed to list models: {e}")
                return False
        else:
            print("❌ No API key found")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import google.generativeai: {e}")
        return False

if __name__ == "__main__":
    success = test_chatbot_connection()
    if success:
        print("\n🎉 CHATBOT API IS WORKING!")
        print("💡 Update your chatbot_service.py to use the working model")
        print("📝 Recommended model name: gemini-pro")
    else:
        print("\n❌ CHATBOT API IS NOT WORKING!")
        print("🔧 Check your API key and internet connection")
