from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .chatbot_service import ChatbotService
from .models import ChatHistory

class ChatbotView(APIView):
    def post(self, request):
        user_message = request.data.get('message', '')
        # Support both JSON and multipart/form-data
        if not user_message and 'message' in request.POST:
            user_message = request.POST.get('message', '')
            
        # Handle optional file upload
        uploaded_file = request.FILES.get('file')
        
        if not user_message and not uploaded_file:
            return Response({"error": "Message or file is required"}, status=status.HTTP_400_BAD_REQUEST)
        
        history = request.data.get('history') or []
        if not history and 'history' in request.POST:
            import json
            try:
                history = json.loads(request.POST.get('history', '[]'))
            except:
                history = []
        
        bot_response = ChatbotService.get_response(user_message, history=history, uploaded_file=uploaded_file)
        
        # Save to history (only text for now)
        ChatHistory.objects.create(
            user_message=user_message,
            bot_response=bot_response
        )
        
        return Response({"response": bot_response}, status=status.HTTP_200_OK)
