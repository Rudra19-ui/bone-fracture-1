import os
try:
    import google.genai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
if HAS_GENAI:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class ChatbotService:
    @staticmethod
    def get_response(user_message, history=None, uploaded_file=None):
        # Basic input sanitation
        if not isinstance(user_message, str) or not user_message.strip():
            if not uploaded_file:
                return "Please enter a message or upload a report. I can help with analysis, healing timelines, and precautions."
            user_message = "Please analyze this report/image and provide medical guidance."

        user_message_lower = user_message.lower().strip()
        history = history or []
        
        # 1. Try Gemini AI first (Recommended Mode)
        api_key = os.getenv("GEMINI_API_KEY")
        if HAS_GENAI and api_key and api_key != "your_gemini_api_key_here" and len(api_key) > 5:
            try:
                # Try different model names in order of preference
                model_names = ["gemini-1.5-flash", "gemini-pro"]
                model = None
                last_error = None
                
                for model_name in model_names:
                    try:
                        model = genai.GenerativeModel(model_name)
                        # Test the model with a simple request to verify it works
                        test_response = model.generate_content("Test connection")
                        if test_response and test_response.text:
                            model = genai.GenerativeModel(model_name)  # Use the working model
                            break
                    except Exception as e:
                        last_error = e
                        continue
                
                if not model:
                    # If all models fail, try without specifying model (let API choose)
                    try:
                        model = genai.GenerativeModel()
                        test_response = model.generate_content("Test connection")
                        if test_response and test_response.text:
                            print(f"✅ Using default model (auto-selected)")
                    except Exception as e:
                        raise Exception(f"All models failed. Last error: {last_error}")
                
                # System context to guide the AI as a professional Medical Advisor
                system_context = (
                    "You are a 'FractureAI Medical Advisor', an advanced AI agent specialized in orthopedic bone health. "
                    "Your tone is professional, empathetic, and highly informative, similar to a senior orthopedic consultant. "
                    "SYSTEM CAPABILITIES:\n"
                    "- You analyze X-ray reports and images using ResNet50 and Gemini AI.\n"
                    "- You provide precise healing timelines based on bone type and fracture severity.\n"
                    "- You suggest specific precautions (e.g., R.I.C.E., weight-bearing status, immobilization advice).\n\n"
                    "INSTRUCTIONS:\n"
                    "1. Always act as a direct advisor to patient/doctor.\n"
                    "2. If a fracture is mentioned or detected in an uploaded file, give a estimated recovery path (e.g., 'Typical healing for a non-displaced wrist fracture is 6-8 weeks').\n"
                    "3. Suggest immediate precautions (e.g., 'Keep the limb elevated', 'Do not attempt to lift heavy objects').\n"
                    "4. Answer any general medical or system question with high intelligence, like Gemini or ChatGPT.\n"
                    "5. MANDATORY: End every response with this disclaimer: 'DISCLAIMER: I am an AI assistant, not a doctor. This analysis is for informational purposes and must be verified by a medical professional.'\n"
                )

                # Build conversation context
                convo_snippets = []
                for turn in history[-8:]:  # Keep last 8 messages for better context
                    sender = (turn.get("sender") or "").strip().lower()
                    text = (turn.get("text") or "").strip()
                    if not text: continue
                    role = "User" if sender == "user" else "Assistant"
                    convo_snippets.append(f"{role}: {text}")
                convo_block = "\n".join(convo_snippets)

                prompt_parts = [
                    f"{system_context}\n\n",
                    f"Conversation History:\n{convo_block}\n\n",
                    f"User Message: {user_message}\n",
                    "Assistant Advisor:"
                ]

                # If a file is uploaded, add it to the request
                if uploaded_file:
                    file_content = uploaded_file.read()
                    mime_type = uploaded_file.content_type
                    
                    # Gemini supports both images and documents
                    parts = [
                        f"Please analyze this medical file ({mime_type}) in the context of our conversation. {user_message}",
                        {"mime_type": mime_type, "data": file_content}
                    ]
                    
                    response = model.generate_content(parts)
                    return response.text

                # Standard text-only response
                response = model.generate_content(prompt_parts)
                return response.text
                
            except Exception as e:
                error_msg = str(e).lower()
                print(f"Gemini API Error: {e}")
                
                # Enhanced error handling
                if "leaked" in error_msg or "403" in error_msg:
                    return "I'm currently experiencing API key authentication issues. Your API key may have been reported as leaked. Please generate a new API key from Google AI Studio and update your configuration. I'll provide assistance using my local knowledge base in the meantime. " + ChatbotService._get_fallback_response(user_message_lower)
                elif "404" in error_msg or "model" in error_msg or "not found" in error_msg:
                    return "I'm currently experiencing model availability issues with the AI service. The AI models may be temporarily unavailable. I'll provide assistance using my local knowledge base in the meantime. " + ChatbotService._get_fallback_response(user_message_lower)
                elif "429" in error_msg or "quota" in error_msg:
                    return "I'm currently experiencing API rate limiting. The service is temporarily unavailable due to high demand. I'll provide assistance using my local knowledge base in the meantime. " + ChatbotService._get_fallback_response(user_message_lower)
                else:
                    return "I'm currently experiencing connectivity issues with my primary AI processing unit. I'll provide assistance using my local knowledge base in the meantime. " + ChatbotService._get_fallback_response(user_message_lower)
        else:
            return "I'm currently operating in limited mode because my API key is not configured or the required package is not installed. I can still answer basic questions about healing and precautions using my local knowledge base. " + ChatbotService._get_fallback_response(user_message_lower)
    
    @staticmethod
    def _get_fallback_response(user_message_lower):
        # Enhanced fallback with more comprehensive medical knowledge
        knowledge_base = {
            "wrist": {
                "keywords": ["wrist", "radius", "ulna", "carpal", "scaphoid"],
                "response": "For a suspected Wrist (Distal Radius/Ulna) fracture: 1. Immediate orthopedic consultation is required for alignment assessment. 2. Immobilize the specific area (e.g., splint/cast). 3. Clinical correlation is mandatory. Typical healing for non-displaced wrist fractures is 6-8 weeks. For displaced fractures, 4-12 weeks may be needed."
            },
            "hand": {
                "keywords": ["hand", "finger", "phalanx", "metacarpal"],
                "response": "For Hand fractures: 1. Immediate immobilization with splint or cast is essential. 2. Avoid any gripping or lifting. 3. Elevate hand above heart level to reduce swelling. 4. Apply ice packs wrapped in cloth. 5. Healing typically takes 3-6 weeks for simple fractures, 6-10 weeks for complex fractures."
            },
            "elbow": {
                "keywords": ["elbow", "humerus", "olecranon", "distal"],
                "response": "For Elbow fractures: 1. Keep elbow at 90 degrees with sling support. 2. Avoid any bending or straightening. 3. Apply ice to reduce swelling. 4. Healing usually requires 6-8 weeks. Physical therapy may be needed for full recovery."
            },
            "shoulder": {
                "keywords": ["shoulder", "clavicle", "acromion", "glenoid"],
                "response": "For Shoulder fractures: 1. Use arm sling for immobilization. 2. Avoid any overhead activities. 3. Apply ice and pain medication as prescribed. 4. Recovery typically takes 8-12 weeks. Surgical intervention may be required for severe fractures."
            },
            "ankle": {
                "keywords": ["ankle", "tibia", "fibula", "malleolus"],
                "response": "For Ankle fractures: 1. Immediate immobilization and non-weight bearing is crucial. 2. Elevate ankle and apply ice. 3. Use crutches for mobility. 4. Healing typically takes 6-12 weeks depending on severity."
            },
            "healing": {
                "keywords": ["healing", "recovery", "how long", "timeline", "weeks", "months"],
                "response": "Bone healing occurs in phases: Inflammation (1-2 weeks), Soft Callus formation (2-6 weeks), and Hard Callus remodeling (6-12 weeks). Most fractures require 6-12 weeks total healing time. Factors affecting healing: age, nutrition, blood supply, fracture stability."
            },
            "precaution": {
                "keywords": ["precaution", "do", "dont", "care", "safe", "prevent", "avoid"],
                "response": "Standard orthopedic precautions include: 1. R.I.C.E. method - Rest, Ice (20 mins on/off), Compression, Elevation. 2. No weight bearing on affected limb. 3. Keep joint immobilized. 4. Follow up with orthopedic specialist as scheduled."
            },
            "pain": {
                "keywords": ["pain", "hurt", "ache", "sore", "uncomfortable"],
                "response": "For pain management: 1. Use prescribed pain medications as directed. 2. Apply ice for swelling and inflammation. 3. Elevate affected limb. 4. Avoid activities that increase pain. 5. Contact doctor if pain is severe or worsening."
            },
            "greetings": {
                "keywords": ["hi", "hello", "hey", "what are you", "help"],
                "response": "Greetings. I am your FractureAI Medical Advisor. I can help with fracture analysis, healing timelines, precautions, and answer medical questions. How can I assist you today?"
            },
            "report": {
                "keywords": ["report", "analyze", "file", "upload", "attached", "x-ray"],
                "response": "I can analyze medical reports and X-ray images. Please upload your report or image, and I'll provide detailed medical guidance based on the findings."
            }
        }

        # Check fallback knowledge base
        for category, data in knowledge_base.items():
            if any(keyword in user_message_lower for keyword in data["keywords"]):
                return f"{data['response']}\n\nDISCLAIMER: I am an AI assistant, not a doctor. This analysis is for informational purposes and must be verified by a medical professional."

        # Default response for unrecognized queries
        return "I can help with questions about bone fractures, healing timelines, precautions, and medical guidance. Please ask about specific bone types (wrist, hand, elbow, shoulder, ankle) or upload your medical report for analysis.\n\nDISCLAIMER: I am an AI assistant, not a doctor. This analysis is for informational purposes and must be verified by a medical professional."
