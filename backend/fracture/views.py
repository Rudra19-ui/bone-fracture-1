from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import ImageAnalysisSerializer
from .models import ImageAnalysis
from .predictions_engine import predict, ANATOMICAL_MAP
import json

class AnalysisView(APIView):
    def post(self, request):
        print("DEBUG: API POST request received")
        image_file = request.FILES.get('image')
        if not image_file:
            print("DEBUG: No image provided")
            return Response({"error": "No image provided"}, status=status.HTTP_400_BAD_REQUEST)

        print(f"DEBUG: Image file received: {image_file.name}")

        # No cache check — always run fresh AI inference to avoid stale misclassifications

        # Perform REAL AI Inference
        # Save image temporarily to disk for processing
        instance = ImageAnalysis(
            image=image_file,
            image_name=request.data.get('image_name', image_file.name),
            user_name=request.data.get('user_name', ''),
            user_type=request.data.get('user_type', '')
        )
        print("DEBUG: ImageAnalysis instance created")
        instance.save() # Saves file and hash
        print("DEBUG: ImageAnalysis instance saved")

        try:
            img_path = instance.image.path
            print(f"DEBUG: Image path: {img_path}")
            
            # 1. Predict Bone Part
            print("DEBUG: Starting bone type prediction...")
            # If user manually selected a bone type from the UI, trust it — skip AI detection
            bone_type_hint = request.data.get('bone_type_hint', '').strip()
            valid_parts = ["Elbow", "Hand", "Shoulder", "Wrist", "Ankle"]
            print(f"DEBUG: Bone type hint: '{bone_type_hint}'")
            
            if bone_type_hint and bone_type_hint in valid_parts:
                bone_type = bone_type_hint
                print(f"DEBUG: Using user-selected bone type: {bone_type}")
            else:
                print("DEBUG: Starting AI bone type detection...")
                # force_fresh=True ensures bone type is always re-detected from image, not from cache
                bone_type = predict(img_path, model="Parts", force_fresh=True)
                print(f"DEBUG: AI detected bone type: {bone_type}")
            
            # 2. Predict Fracture
            print("DEBUG: Starting fracture prediction...")
            fracture_result = predict(img_path, model=bone_type)
            print(f"DEBUG: Fracture prediction result: {fracture_result}")
            
            # Update instance with real results
            if isinstance(fracture_result, dict):
                # Use accurate bone type from fracture analysis if available, otherwise use parts detection
                detected_bone_type = fracture_result.get('bone_type', bone_type)
                instance.bone_type = detected_bone_type
                instance.fracture_detected = (fracture_result['result'] == "DETECTED")
                instance.confidence = fracture_result.get('probability', 0.0) * 100
                instance.severity = "High Risk" if instance.fracture_detected else "Normal"
                instance.location = fracture_result.get('location', ANATOMICAL_MAP.get(detected_bone_type, 'Bone Structure'))
                instance.report_data = fracture_result
                print("DEBUG: Instance updated with fracture results")
            else:
                # Fallback: ensure bone_type is set from parts detection
                instance.bone_type = bone_type
                instance.location = ANATOMICAL_MAP.get(bone_type, 'Bone Structure')
                print("DEBUG: Instance updated with fallback results")
            
            print("DEBUG: Saving instance...")
            instance.save()
            print("DEBUG: Instance saved successfully")
            
            print("DEBUG: Serializing response...")
            serialized_data = ImageAnalysisSerializer(instance).data
            print("DEBUG: Response serialized")
            
            return Response(serialized_data, status=status.HTTP_201_CREATED)

        except Exception as e:
            return Response({"error": f"Inference failed: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def get(self, request):
        print("DEBUG: API GET request received")
        image_name = request.query_params.get('image_name')
        image_hash = request.query_params.get('image_hash')
        if image_hash:
            obj = ImageAnalysis.objects.filter(image_hash=image_hash).order_by('-uploaded_at').first()
            if obj:
                return Response(ImageAnalysisSerializer(obj).data)
            return Response({}, status=status.HTTP_200_OK)
        if image_name:
            obj = ImageAnalysis.objects.filter(image_name=image_name).order_by('-uploaded_at').first()
            if obj:
                return Response(ImageAnalysisSerializer(obj).data)
            return Response({}, status=status.HTTP_200_OK)
        qs = ImageAnalysis.objects.order_by('-uploaded_at')[:20]
        return Response(ImageAnalysisSerializer(qs, many=True).data)