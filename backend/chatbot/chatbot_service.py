import os
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API
if HAS_GEMINI:
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
        if HAS_GEMINI and api_key and api_key != "your_gemini_api_key_here" and len(api_key) > 5:
            try:
                # Use gemini-2.0-flash for speed and multi-modal support
                model = genai.GenerativeModel("gemini-2.0-flash")
                
                # System context to guide the AI as a professional Medical Advisor
                system_context = (
                    "You are the 'FractureAI Medical Advisor', an advanced AI agent specialized in orthopedic bone health. "
                    "Your tone is professional, empathetic, and highly informative, similar to a senior orthopedic consultant. "
                    "SYSTEM CAPABILITIES:\n"
                    "- You analyze X-ray reports and images using ResNet50 and Gemini AI.\n"
                    "- You provide precise healing timelines based on the bone type and fracture severity.\n"
                    "- You suggest specific precautions (e.g., R.I.C.E., weight-bearing status, immobilization advice).\n\n"
                    "INSTRUCTIONS:\n"
                    "1. Always act as a direct advisor to the patient/doctor.\n"
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

                # If a file is uploaded, add it to the request (Image or PDF)
                if uploaded_file:
                    file_content = uploaded_file.read()
                    mime_type = uploaded_file.content_type
                    
                    # Gemini 1.5 Flash supports both images and documents (PDF)
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
                print(f"Gemini API Error (Advisor Mode): {e}")
                return "I'm currently experiencing connectivity issues with my primary AI processing unit. I'll provide assistance using my local knowledge base in the meantime. " + self._get_fallback_response(user_message_lower)
        else:
            return "I'm currently operating in limited mode because my API key is not configured. I can still answer basic questions about healing and precautions using my local knowledge base. " + self._get_fallback_response(user_message_lower)
    
    @staticmethod
    def _get_fallback_response(user_message_lower):
        # 2. Intelligent Fallback Logic (if AI/API fails)
        knowledge_base = {
            "wrist": {
                "keywords": ["wrist", "radius", "ulna"],
                "response": "For a suspected Wrist (Distal Radius/Ulna) fracture: 1. Immediate orthopedic consultation is required for alignment assessment. 2. Immobilize the specific area (e.g., splint/cast). 3. Clinical correlation is mandatory. Typical healing for non-displaced wrist fractures is 6-8 weeks."
            },
            "healing": {
                "keywords": ["month", "week", "long", "healing", "recovery", "time"],
                "response": "Bone healing typically occurs in phases: Inflammation (1-2 weeks), Soft Callus (2-6 weeks), and Hard Callus (6-12 weeks). Most simple fractures require 6-12 weeks of total recovery time depending on the bone type and severity."
            },
            "precaution": {
                "keywords": ["precaution", "do", "dont", "care", "safe", "prevent"],
                "response": "Standard orthopedic precautions include the R.I.C.E method: Rest, Ice (20 mins on/off), Compression (moderate), and Elevation (above heart level). Avoid weight-bearing on the affected limb until cleared by a professional."
            },
            "greetings": {
                "keywords": ["hi", "hello", "hey", "what are you doing"],
                "response": "Greetings. I am your FractureAI Medical Advisor. I am here to help you understand your reports or discuss precautions for various bone fractures. How can I help you?"
            },
            "report": {
                "keywords": ["report", "analyze", "file", "upload", "attached"],
                "response": "I see you've uploaded a report. In Limited Mode, I cannot perform live vision analysis. However, based on typical fracture reports: 1. Seek an orthopedic consultation. 2. Immobilize the joint. 3. Follow the 'Next Steps' in your report exactly. If you tell me the bone type, I can provide more specific healing info."
            }
        }

        # Check fallback KB
        for category, data in knowledge_base.items():
            if any(keyword in user_message_lower for keyword in data["keywords"]):
                return f"{data['response']}\n\nDISCLAIMER: I am an AI assistant, not a doctor. Results must be verified by a medical professional."

        return "Please ask about healing times, precautions, or specific bone types (e.g., Wrist, Elbow, Hand). For a full analysis, please ensure the Gemini API key is properly configured."
