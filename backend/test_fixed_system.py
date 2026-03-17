#!/usr/bin/env python3
"""
Test Script for Fixed Bone Fracture Detection System
Tests the clean prediction pipeline with multiple images
"""

import os
import sys
import json
from datetime import datetime

# Add backend path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_prediction_system():
    """Test the fixed prediction system with sample images"""
    
    print("=" * 60)
    print("🦴 BONE FRACTURE DETECTION - FIXED SYSTEM TEST")
    print("=" * 60)
    
    try:
        # Import the fixed prediction engine
        from fracture.predictions_engine_fixed import predict_bone_type, predict_fracture
        print("✅ Fixed prediction engine imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import fixed prediction engine: {e}")
        return
    
    # Test images directory
    test_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'Bone-Fracture-Detection-master', 'Dataset', 'train_valid')
    
    if not os.path.exists(test_dir):
        print(f"❌ Test directory not found: {test_dir}")
        print("📁 Available directories:")
        parent_dir = os.path.dirname(test_dir)
        if os.path.exists(parent_dir):
            for item in os.listdir(parent_dir):
                print(f"   - {item}")
        return
    
    # Find test images
    test_images = []
    bone_types = ['Elbow', 'Hand', 'Shoulder']
    
    for bone_type in bone_types:
        bone_dir = os.path.join(test_dir, bone_type)
        if os.path.exists(bone_dir):
            # Find first patient with images
            for patient in os.listdir(bone_dir)[:3]:  # Test first 3 patients
                patient_dir = os.path.join(bone_dir, patient)
                if os.path.isdir(patient_dir):
                    for study in os.listdir(patient_dir):
                        study_dir = os.path.join(patient_dir, study)
                        if os.path.isdir(study_dir):
                            for img_file in os.listdir(study_dir)[:2]:  # Test 2 images per study
                                img_path = os.path.join(study_dir, img_file)
                                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                    # Determine actual label from study name
                                    actual_label = "fractured" if "positive" in study.lower() else "normal"
                                    test_images.append({
                                        'path': img_path,
                                        'bone_type': bone_type,
                                        'actual_label': actual_label,
                                        'filename': img_file
                                    })
    
    if not test_images:
        print("❌ No test images found")
        return
    
    print(f"📊 Found {len(test_images)} test images")
    print("-" * 60)
    
    # Test predictions
    results = []
    correct_predictions = 0
    total_predictions = 0
    
    for i, test_img in enumerate(test_images[:10]):  # Test first 10 images
        print(f"\n🔍 Test {i+1}/{min(len(test_images), 10)}")
        print(f"📁 Image: {test_img['filename']}")
        print(f"🦴 Bone Type: {test_img['bone_type']}")
        print(f"🏷️  Actual Label: {test_img['actual_label']}")
        
        try:
            # Test bone type prediction
            predicted_bone_type = predict_bone_type(test_img['path'], force_fresh=True)
            print(f"🔬 Predicted Bone Type: {predicted_bone_type}")
            
            # Test fracture prediction
            fracture_result = predict_fracture(test_img['path'], predicted_bone_type, force_fresh=True)
            
            if fracture_result.get('result') == 'ERROR':
                print(f"❌ Prediction failed: {fracture_result.get('error', 'Unknown error')}")
                continue
            
            predicted_label = "fractured" if fracture_result['fracture_detected'] else "normal"
            confidence = fracture_result['probability']
            
            print(f"🔬 Predicted Label: {predicted_label}")
            print(f"📊 Confidence: {confidence:.3f}")
            print(f"📈 Model Probabilities: {fracture_result.get('model_probabilities', {})}")
            
            # Check if prediction is correct
            is_correct = predicted_label == test_img['actual_label']
            if is_correct:
                correct_predictions += 1
                print("✅ CORRECT")
            else:
                print("❌ INCORRECT")
            
            total_predictions += 1
            
            # Store result
            results.append({
                'filename': test_img['filename'],
                'bone_type': test_img['bone_type'],
                'actual_label': test_img['actual_label'],
                'predicted_bone_type': predicted_bone_type,
                'predicted_label': predicted_label,
                'confidence': confidence,
                'correct': is_correct,
                'result': fracture_result['result'],
                'error': None
            })
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            results.append({
                'filename': test_img['filename'],
                'bone_type': test_img['bone_type'],
                'actual_label': test_img['actual_label'],
                'predicted_bone_type': None,
                'predicted_label': None,
                'confidence': 0.0,
                'correct': False,
                'result': 'ERROR',
                'error': str(e)
            })
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"✅ Total Predictions: {total_predictions}")
        print(f"✅ Correct Predictions: {correct_predictions}")
        print(f"📈 Accuracy: {accuracy:.1f}%")
        
        # Breakdown by actual label
        fracture_correct = sum(1 for r in results if r['actual_label'] == 'fractured' and r['correct'])
        fracture_total = sum(1 for r in results if r['actual_label'] == 'fractured')
        normal_correct = sum(1 for r in results if r['actual_label'] == 'normal' and r['correct'])
        normal_total = sum(1 for r in results if r['actual_label'] == 'normal')
        
        print(f"\n🔍 Detailed Breakdown:")
        if fracture_total > 0:
            print(f"   Fracture Detection: {fracture_correct}/{fracture_total} ({(fracture_correct/fracture_total)*100:.1f}%)")
        if normal_total > 0:
            print(f"   Normal Detection: {normal_correct}/{normal_total} ({(normal_correct/normal_total)*100:.1f}%)")
    else:
        print("❌ No successful predictions made")
    
    # Save detailed results
    results_file = os.path.join(os.path.dirname(__file__), 'test_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(test_images),
            'successful_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'accuracy': (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0,
            'results': results
        }, f, indent=2)
    
    print(f"\n📁 Detailed results saved to: {results_file}")
    print("=" * 60)

def test_label_mapping():
    """Test and verify label mapping"""
    print("\n🏷️  LABEL MAPPING VERIFICATION")
    print("-" * 30)
    
    try:
        from fracture.predictions_engine_fixed import categories_fracture
        print(f"Fracture categories: {categories_fracture}")
        print(f"Index 0: {categories_fracture[0]} (should be 'fractured')")
        print(f"Index 1: {categories_fracture[1]} (should be 'normal')")
        
        if categories_fracture[0] == 'fractured' and categories_fracture[1] == 'normal':
            print("✅ Label mapping is CORRECT")
        else:
            print("❌ Label mapping is INCORRECT")
            
    except Exception as e:
        print(f"❌ Label mapping test failed: {e}")

def test_model_loading():
    """Test model loading functionality"""
    print("\n🏗️  MODEL LOADING TEST")
    print("-" * 30)
    
    try:
        from fracture.predictions_engine_fixed import get_model
        import tensorflow as tf
        
        print("Testing model loading...")
        
        # Test loading each model
        models_to_test = ["Parts", "Elbow", "Hand", "Shoulder"]
        
        for model_name in models_to_test:
            try:
                print(f"Loading {model_name} model...")
                model = get_model(model_name)
                print(f"✅ {model_name} model loaded successfully")
                print(f"   Input shape: {model.input_shape}")
                print(f"   Output shape: {model.output_shape}")
                
                # Clear session for next model
                tf.keras.backend.clear_session()
                
            except Exception as e:
                print(f"❌ Failed to load {model_name} model: {e}")
                
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting Fixed Bone Fracture Detection System Tests")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run tests
    test_label_mapping()
    test_model_loading()
    test_prediction_system()
    
    print(f"\n⏰ Tests completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎉 All tests completed!")
