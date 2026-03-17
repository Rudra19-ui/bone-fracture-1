"""
Django API endpoint for AI Medical Assistant
Handles PDF upload and medical analysis
"""

import os
import json
from datetime import datetime
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.files.storage import default_storage
from django.conf import settings
from .medical_assistant import FractureReportAnalyzer

@csrf_exempt
@require_http_methods(["POST"])
def medical_analysis_view(request):
    """
    API endpoint for medical PDF analysis
    
    POST /api/medical-analysis/
    Content-Type: multipart/form-data
    Body: pdf_file (file)
    
    Returns:
    {
        "timestamp": "2026-03-17T...",
        "report_summary": {...},
        "interpretation": "...",
        "severity_insight": "...",
        "recommended_actions": [...],
        "recovery_expectation": "...",
        "additional_advice": [...],
        "disclaimer": "...",
        "extracted_data": {...}
    }
    """
    try:
        # Validate PDF file
        if 'pdf_file' not in request.FILES:
            return JsonResponse({
                "error": "No PDF file provided",
                "message": "Please upload a PDF file for analysis"
            }, status=400)
        
        pdf_file = request.FILES['pdf_file']
        
        # Validate file type
        if not pdf_file.name.lower().endswith('.pdf'):
            return JsonResponse({
                "error": "Invalid file type",
                "message": "Please upload a PDF file"
            }, status=400)
        
        # Validate file size (max 10MB)
        if pdf_file.size > 10 * 1024 * 1024:
            return JsonResponse({
                "error": "File too large",
                "message": "PDF file must be less than 10MB"
            }, status=400)
        
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"medical_analysis_{timestamp}_{pdf_file.name}"
        
        # Save file temporarily
        file_path = default_storage.save(f"temp/{filename}", pdf_file)
        full_path = os.path.join(settings.MEDIA_ROOT, file_path)
        
        try:
            # Initialize analyzer
            analyzer = FractureReportAnalyzer()
            
            # Analyze PDF
            result = analyzer.analyze_pdf_report(full_path)
            
            # Check for analysis errors
            if "error" in result:
                return JsonResponse({
                    "error": "Analysis failed",
                    "message": result["error"]
                }, status=500)
            
            # Add metadata
            result["analysis_metadata"] = {
                "filename": pdf_file.name,
                "file_size": pdf_file.size,
                "analysis_timestamp": result["timestamp"],
                "analysis_version": "1.0"
            }
            
            return JsonResponse(result, status=200)
            
        finally:
            # Clean up temporary file
            if default_storage.exists(file_path):
                default_storage.delete(file_path)
    
    except Exception as e:
        return JsonResponse({
            "error": "Internal server error",
            "message": str(e)
        }, status=500)

@csrf_exempt
@require_http_methods(["GET"])
def medical_analysis_status_view(request):
    """
    API endpoint to check medical analysis service status
    
    GET /api/medical-analysis/status/
    
    Returns:
    {
        "status": "active",
        "services": {
            "pdf_processing": true,
            "medical_rules": true,
            "text_extraction": "PyMuPDF" or "pdfplumber"
        },
        "timestamp": "2026-03-17T..."
    }
    """
    try:
        analyzer = FractureReportAnalyzer()
        
        # Check text extraction capability
        text_extraction = "None"
        try:
            import fitz
            text_extraction = "PyMuPDF"
        except ImportError:
            try:
                import pdfplumber
                text_extraction = "pdfplumber"
            except ImportError:
                pass
        
        status = {
            "status": "active",
            "services": {
                "pdf_processing": True,
                "medical_rules": bool(analyzer.medical_rules),
                "text_extraction": text_extraction
            },
            "timestamp": datetime.now().isoformat(),
            "version": "1.0"
        }
        
        return JsonResponse(status)
    
    except Exception as e:
        return JsonResponse({
            "status": "error",
            "message": str(e)
        }, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def medical_analysis_text_view(request):
    """
    API endpoint for text-based medical analysis (for testing)
    
    POST /api/medical-analysis/text/
    Content-Type: application/json
    Body: {
        "text": "Fracture detected: YES, Bone: Hand, Confidence: 95%"
    }
    
    Returns same format as PDF analysis
    """
    try:
        # Parse JSON body
        try:
            body = json.loads(request.body)
            text = body.get('text', '')
        except json.JSONDecodeError:
            return JsonResponse({
                "error": "Invalid JSON",
                "message": "Please provide valid JSON data"
            }, status=400)
        
        if not text:
            return JsonResponse({
                "error": "No text provided",
                "message": "Please provide text for analysis"
            }, status=400)
        
        # Initialize analyzer
        analyzer = FractureReportAnalyzer()
        
        # Parse fracture data from text
        data = analyzer.parse_fracture_data(text)
        
        # Generate medical response
        result = analyzer.generate_medical_response(data)
        
        # Add metadata
        result["analysis_metadata"] = {
            "input_type": "text",
            "analysis_timestamp": result["timestamp"],
            "analysis_version": "1.0"
        }
        
        return JsonResponse(result, status=200)
    
    except Exception as e:
        return JsonResponse({
            "error": "Internal server error",
            "message": str(e)
        }, status=500)

# Example URL configuration for urls.py
def get_medical_analysis_urls():
    """
    Returns URL patterns for medical analysis endpoints
    Add this to your urls.py:
    
    from .medical_analysis_api import get_medical_analysis_urls
    urlpatterns += get_medical_analysis_urls()
    """
    from django.urls import path
    
    urlpatterns = [
        path('medical-analysis/', medical_analysis_view, name='medical_analysis'),
        path('medical-analysis/status/', medical_analysis_status_view, name='medical_analysis_status'),
        path('medical-analysis/text/', medical_analysis_text_view, name='medical_analysis_text'),
    ]
    
    return urlpatterns
