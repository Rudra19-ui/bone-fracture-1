"""
AI Medical Assistant for Bone Fracture Analysis
Processes PDF reports and provides medical guidance
"""

import os
import json
import re
from datetime import datetime
from typing import Dict, Optional, Tuple

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    try:
        import pdfplumber
        HAS_PDFPLUMBER = True
    except ImportError:
        HAS_PDFPLUMBER = False

class FractureReportAnalyzer:
    """
    Analyzes bone fracture PDF reports and provides medical guidance
    """
    
    def __init__(self):
        self.medical_rules = self._load_medical_rules()
    
    def _load_medical_rules(self) -> Dict:
        """Load predefined medical guidance rules"""
        return {
            "fracture": {
                "general": {
                    "actions": [
                        "Immediate immobilization",
                        "Avoid movement", 
                        "Consult orthopedic doctor"
                    ],
                    "severity_keywords": ["displacement", "severe", "compound", "comminuted"]
                },
                "hand": {
                    "actions": [
                        "Immobilize using splint or cast",
                        "Avoid gripping or lifting",
                        "Elevate hand to reduce swelling",
                        "Apply ice (wrapped)"
                    ],
                    "recovery_time": {
                        "minor": "3-6 weeks",
                        "severe": "6-10 weeks"
                    }
                },
                "elbow": {
                    "actions": [
                        "Keep elbow immobilized at 90 degrees",
                        "Avoid bending or pressure",
                        "Use sling support"
                    ],
                    "recovery_time": "6-8 weeks"
                },
                "shoulder": {
                    "actions": [
                        "Use arm sling",
                        "Avoid overhead movement",
                        "Strict rest required"
                    ],
                    "recovery_time": "8-12 weeks"
                }
            },
            "normal": {
                "actions": [
                    "Light rest if pain exists",
                    "Optional doctor consultation",
                    "Monitor symptoms"
                ],
                "reassurance": "This is a positive result, no fracture pattern found."
            }
        }
    
    def extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        """Extract text from PDF file"""
        if not os.path.exists(pdf_path):
            return None
        
        text = ""
        
        # Try PyMuPDF first (faster)
        if HAS_PYMUPDF:
            try:
                doc = fitz.open(pdf_path)
                for page in doc:
                    text += page.get_text()
                doc.close()
                return text
            except Exception as e:
                print(f"PyMuPDF failed: {e}")
        
        # Fallback to pdfplumber
        if HAS_PDFPLUMBER:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        text += page.extract_text() or ""
                return text
            except Exception as e:
                print(f"pdfplumber failed: {e}")
        
        return None
    
    def parse_fracture_data(self, text: str) -> Dict:
        """Parse fracture data from extracted text"""
        data = {
            "fracture_detected": "UNKNOWN",
            "bone_type": "UNKNOWN",
            "confidence": 0,
            "location": "UNKNOWN",
            "status": "UNKNOWN",
            "pattern_suggestion": "UNKNOWN",
            "next_steps": []
        }
        
        # Convert to lowercase for easier parsing
        text_lower = text.lower()
        
        # Extract fracture detection
        fracture_indicators = ["fracture detected", "fracture present", "bone fracture", "detected"]
        normal_indicators = ["no fracture", "normal", "no fracture pattern", "negative"]
        
        if any(indicator in text_lower for indicator in fracture_indicators):
            data["fracture_detected"] = "YES"
            data["status"] = "DETECTED"
        elif any(indicator in text_lower for indicator in normal_indicators):
            data["fracture_detected"] = "NO"
            data["status"] = "NORMAL"
        
        # Extract bone type
        bone_types = ["hand", "elbow", "shoulder", "wrist", "ankle", "finger", "humerus", "radius", "ulna"]
        for bone in bone_types:
            if bone in text_lower:
                data["bone_type"] = bone.capitalize()
                break
        
        # Extract confidence percentage
        confidence_match = re.search(r'(\d+)%', text)
        if confidence_match:
            data["confidence"] = int(confidence_match.group(1))
        
        # Extract location
        location_patterns = [
            r'location[:\s]*([^\n]+)',
            r'region[:\s]*([^\n]+)',
            r'area[:\s]*([^\n]+)'
        ]
        for pattern in location_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data["location"] = match.group(1).strip()
                break
        
        # Extract pattern suggestion
        pattern_patterns = [
            r'pattern[:\s]*([^\n]+)',
            r'suggestion[:\s]*([^\n]+)',
            r'finding[:\s]*([^\n]+)'
        ]
        for pattern in pattern_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data["pattern_suggestion"] = match.group(1).strip()
                break
        
        # Extract next steps/recommendations
        next_steps_patterns = [
            r'recommendation[:\s]*([^\n]+)',
            r'next step[:\s]*([^\n]+)',
            r'action[:\s]*([^\n]+)'
        ]
        for pattern in next_steps_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                data["next_steps"] = [match.strip() for match in matches]
                break
        
        return data
    
    def determine_severity(self, data: Dict) -> str:
        """Determine severity based on pattern and other factors"""
        if data["fracture_detected"] != "YES":
            return "NORMAL"
        
        pattern = data.get("pattern_suggestion", "").lower()
        confidence = data.get("confidence", 0)
        
        # Check for severity keywords
        severity_keywords = self.medical_rules["fracture"]["general"]["severity_keywords"]
        if any(keyword in pattern for keyword in severity_keywords):
            return "HIGH"
        elif confidence < 70:
            return "MODERATE"
        elif "unclear" in pattern or "inconclusive" in pattern:
            return "MODERATE"
        else:
            return "MODERATE"
    
    def generate_medical_response(self, data: Dict) -> Dict:
        """Generate structured medical response"""
        severity = self.determine_severity(data)
        is_fracture = data["fracture_detected"] == "YES"
        
        response = {
            "timestamp": datetime.now().isoformat(),
            "report_summary": self._generate_report_summary(data),
            "interpretation": self._generate_interpretation(data),
            "severity_insight": self._generate_severity_insight(severity, data),
            "recommended_actions": self._generate_recommended_actions(data, severity),
            "recovery_expectation": self._generate_recovery_expectation(data, severity),
            "additional_advice": self._generate_additional_advice(data, severity),
            "disclaimer": "This AI analysis is for informational purposes only and is not a medical diagnosis. Please consult a qualified healthcare professional.",
            "extracted_data": data
        }
        
        return response
    
    def _generate_report_summary(self, data: Dict) -> Dict:
        """Generate report summary"""
        summary = {
            "fracture_detected": data["fracture_detected"],
            "bone_type": data["bone_type"],
            "confidence": data["confidence"]
        }
        
        if data["location"] != "UNKNOWN":
            summary["location"] = data["location"]
        
        if data["status"] != "UNKNOWN":
            summary["status"] = data["status"]
        
        return summary
    
    def _generate_interpretation(self, data: Dict) -> str:
        """Generate interpretation"""
        if data["fracture_detected"] == "YES":
            bone = data["bone_type"]
            confidence = data["confidence"]
            return f"A fracture has been detected in the {bone.lower()} with {confidence}% confidence."
        else:
            return "No fracture has been detected."
    
    def _generate_severity_insight(self, severity: str, data: Dict) -> str:
        """Generate severity insight"""
        if severity == "NORMAL":
            return "No fracture detected - this is a normal result."
        elif severity == "HIGH":
            return "Signs of bone displacement suggest this may be a serious fracture."
        elif severity == "MODERATE":
            if data["confidence"] < 70:
                return "Low confidence result, further medical evaluation is recommended."
            else:
                return "Fracture detected, requires medical attention."
        else:
            return "Further evaluation needed."
    
    def _generate_recommended_actions(self, data: Dict, severity: str) -> list:
        """Generate recommended actions"""
        actions = []
        
        if data["fracture_detected"] == "YES":
            bone_type = data["bone_type"].lower()
            
            # General fracture actions
            actions.extend(self.medical_rules["fracture"]["general"]["actions"])
            
            # Bone-specific actions
            if bone_type in self.medical_rules["fracture"]:
                actions.extend(self.medical_rules["fracture"][bone_type]["actions"])
            
            # Severity-specific actions
            if severity == "HIGH":
                actions.insert(0, "Seek emergency medical attention immediately")
            elif severity == "MODERATE" and data["confidence"] < 70:
                actions.append("Consider second opinion due to low confidence")
        else:
            actions.extend(self.medical_rules["normal"]["actions"])
        
        return actions
    
    def _generate_recovery_expectation(self, data: Dict, severity: str) -> str:
        """Generate recovery expectation"""
        if data["fracture_detected"] != "YES":
            return "No recovery needed as no fracture was detected."
        
        bone_type = data["bone_type"].lower()
        
        if bone_type in self.medical_rules["fracture"]:
            bone_rules = self.medical_rules["fracture"][bone_type]
            
            if isinstance(bone_rules.get("recovery_time"), dict):
                # Handle minor/severe recovery times
                if severity == "HIGH":
                    return f"Typically {bone_rules['recovery_time']['severe']} depending on severity"
                else:
                    return f"Typically {bone_rules['recovery_time']['minor']} depending on severity"
            else:
                return f"Typically {bone_rules['recovery_time']}"
        
        return "Recovery time varies depending on severity and treatment."
    
    def _generate_additional_advice(self, data: Dict, severity: str) -> list:
        """Generate additional advice"""
        advice = []
        
        if data["fracture_detected"] == "YES":
            bone_type = data["bone_type"].lower()
            
            if bone_type == "hand":
                advice.extend(["Keep hand elevated", "Apply ice packs", "Avoid lifting"])
            elif bone_type == "elbow":
                advice.extend(["Use arm sling", "Avoid carrying heavy objects"])
            elif bone_type == "shoulder":
                advice.extend(["Strict rest", "Avoid overhead activities"])
            
            if severity == "HIGH":
                advice.append("Follow up with orthopedic specialist within 24-48 hours")
            
            if data["confidence"] < 70:
                advice.append("Consider additional imaging for confirmation")
        else:
            advice.append(self.medical_rules["normal"]["reassurance"])
            if data.get("next_steps"):
                advice.extend(data["next_steps"])
        
        return advice
    
    def analyze_pdf_report(self, pdf_path: str) -> Optional[Dict]:
        """Main method to analyze PDF report"""
        try:
            # Extract text from PDF
            text = self.extract_text_from_pdf(pdf_path)
            if not text:
                return {"error": "Could not extract text from PDF"}
            
            # Parse fracture data
            data = self.parse_fracture_data(text)
            
            # Generate medical response
            response = self.generate_medical_response(data)
            
            return response
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

# Django view integration example
def create_medical_analysis_view():
    """
    Example Django view for medical analysis
    """
    from django.http import JsonResponse
    from django.views.decorators.csrf import csrf_exempt
    from django.views.decorators.http import require_http_methods
    import json
    
    @csrf_exempt
    @require_http_methods(["POST"])
    def medical_analysis_view(request):
        try:
            # Get PDF file from request
            if 'pdf_file' not in request.FILES:
                return JsonResponse({"error": "No PDF file provided"}, status=400)
            
            pdf_file = request.FILES['pdf_file']
            
            # Save temporary file
            temp_path = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            with open(temp_path, 'wb+') as destination:
                for chunk in pdf_file.chunks():
                    destination.write(chunk)
            
            # Analyze PDF
            analyzer = FractureReportAnalyzer()
            result = analyzer.analyze_pdf_report(temp_path)
            
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return JsonResponse(result)
            
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    
    return medical_analysis_view

# Usage example
if __name__ == "__main__":
    # Example usage
    analyzer = FractureReportAnalyzer()
    
    # Analyze a PDF file
    result = analyzer.analyze_pdf_report("fracture-analysis-2026-03-17.pdf")
    print(json.dumps(result, indent=2))
