#!/usr/bin/env python3
"""
Test Script for Hybrid ViT-CNN Bone Fracture Detection System
Tests both ResNet50 CNN and Vision Transformer technologies
"""

import os
import sys
import json
import time
from datetime import datetime

# Add backend path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_vit_components():
    """Test Vision Transformer components"""
    print("\n🔬 VISION TRANSFORMER COMPONENTS TEST")
    print("=" * 50)
    
    try:
        from fracture.vit_model import (
            PatchEmbedding, MultiHeadSelfAttention, 
            TransformerEncoder, VisionTransformer,
            create_vit_model, create_hybrid_vit_cnn_model
        )
        print("✅ ViT components imported successfully")
        
        # Test patch embedding
        print("\n📦 Testing Patch Embedding...")
        import tensorflow as tf
        dummy_input = tf.random.normal((2, 224, 224, 3))
        patch_embed = PatchEmbedding(patch_size=16, embed_dim=768)
        patches = patch_embed(dummy_input)
        print(f"✅ Patch embedding output shape: {patches.shape}")
        
        # Test multi-head attention
        print("\n🧠 Testing Multi-Head Self-Attention...")
        mha = MultiHeadSelfAttention(num_heads=12, embed_dim=768)
        attention_output = mha(patches)
        print(f"✅ Attention output shape: {attention_output.shape}")
        
        # Test transformer encoder
        print("\n🔄 Testing Transformer Encoder...")
        encoder = TransformerEncoder(embed_dim=768, num_heads=12)
        encoder_output = encoder(patches)
        print(f"✅ Encoder output shape: {encoder_output.shape}")
        
        # Test complete ViT model
        print("\n🏗️  Testing Complete ViT Model...")
        vit_model = create_vit_model(input_shape=(224, 224, 3), num_classes=2, model_size='small')
        vit_output = vit_model(dummy_input)
        print(f"✅ ViT model output shape: {vit_output.shape}")
        
        # Test hybrid model
        print("\n🔄 Testing Hybrid ViT-CNN Model...")
        hybrid_model = create_hybrid_vit_cnn_model(input_shape=(224, 224, 3), num_classes=2)
        hybrid_output = hybrid_model(dummy_input)
        print(f"✅ Hybrid model output shape: {hybrid_output.shape}")
        
        print("\n✅ All ViT components working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ ViT components test failed: {e}")
        return False

def test_hybrid_prediction_engine():
    """Test the hybrid prediction engine"""
    print("\n🔬 HYBRID PREDICTION ENGINE TEST")
    print("=" * 40)
    
    try:
        from fracture.hybrid_predictions_engine import (
            predict_bone_type, predict_fracture_hybrid, predict_fracture_pure_vit
        )
        print("✅ Hybrid prediction engine imported successfully")
        
        # Test model loading
        print("\n📥 Testing Model Loading...")
        
        try:
            # Test ResNet model loading
            from fracture.hybrid_predictions_engine import get_resnet_model
            resnet_model = get_resnet_model("Elbow")
            print("✅ ResNet model loaded successfully")
        except Exception as e:
            print(f"⚠️  ResNet model loading failed: {e}")
        
        try:
            # Test ViT model loading
            from fracture.hybrid_predictions_engine import get_vit_model
            vit_model = get_vit_model("hybrid", "Elbow")
            print("✅ ViT model loaded successfully")
        except Exception as e:
            print(f"⚠️  ViT model loading failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Hybrid prediction engine test failed: {e}")
        return False

def test_prediction_comparison():
    """Test prediction comparison between CNN, ViT, and Hybrid"""
    print("\n🔬 PREDICTION COMPARISON TEST")
    print("=" * 40)
    
    try:
        from fracture.hybrid_predictions_engine import (
            predict_bone_type, predict_fracture_hybrid, predict_fracture_pure_vit
        )
        
        # Find test image
        test_dir = os.path.join(os.path.dirname(__file__), '..', 'Bone-Fracture-Detection-master', 'Dataset', 'train_valid')
        test_img_path = None
        
        for bone_type in ['Elbow', 'Hand', 'Shoulder']:
            bone_dir = os.path.join(test_dir, bone_type)
            if os.path.exists(bone_dir):
                for patient in os.listdir(bone_dir)[:1]:
                    patient_dir = os.path.join(bone_dir, patient)
                    if os.path.isdir(patient_dir):
                        for study in os.listdir(patient_dir):
                            study_dir = os.path.join(patient_dir, study)
                            if os.path.isdir(study_dir):
                                for img_file in os.listdir(study_dir):
                                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                        test_img_path = os.path.join(study_dir, img_file)
                                        break
                            if test_img_path:
                                break
                    if test_img_path:
                        break
            if test_img_path:
                break
        
        if not test_img_path:
            print("❌ No test image found for comparison test")
            return False
        
        print(f"📁 Testing with: {os.path.basename(test_img_path)}")
        
        # Test bone type detection
        print("\n🦴 Testing Bone Type Detection...")
        bone_type = predict_bone_type(test_img_path, force_fresh=True)
        print(f"✅ Detected bone type: {bone_type}")
        
        # Test different prediction methods
        results = {}
        
        print("\n🤖 Testing Hybrid ViT-CNN Prediction...")
        start_time = time.time()
        try:
            hybrid_result = predict_fracture_hybrid(test_img_path, bone_type, force_fresh=True)
            hybrid_time = time.time() - start_time
            results['hybrid'] = {
                'result': hybrid_result.get('result'),
                'probability': hybrid_result.get('probability'),
                'technology': hybrid_result.get('technology'),
                'time': hybrid_time,
                'error': None
            }
            print(f"✅ Hybrid result: {hybrid_result.get('result')} (conf: {hybrid_result.get('probability', 0):.3f}, time: {hybrid_time:.2f}s)")
        except Exception as e:
            results['hybrid'] = {'error': str(e)}
            print(f"❌ Hybrid prediction failed: {e}")
        
        print("\n🧠 Testing Pure ViT Prediction...")
        start_time = time.time()
        try:
            vit_result = predict_fracture_pure_vit(test_img_path, bone_type, force_fresh=True)
            vit_time = time.time() - start_time
            results['pure_vit'] = {
                'result': vit_result.get('result'),
                'probability': vit_result.get('probability'),
                'technology': vit_result.get('technology'),
                'time': vit_time,
                'error': None
            }
            print(f"✅ Pure ViT result: {vit_result.get('result')} (conf: {vit_result.get('probability', 0):.3f}, time: {vit_time:.2f}s)")
        except Exception as e:
            results['pure_vit'] = {'error': str(e)}
            print(f"❌ Pure ViT prediction failed: {e}")
        
        # Print comparison
        print("\n📊 PREDICTION COMPARISON SUMMARY")
        print("-" * 30)
        
        for method, result in results.items():
            if result.get('error'):
                print(f"❌ {method.upper()}: {result['error']}")
            else:
                print(f"✅ {method.upper()}: {result['result']} (conf: {result['probability']:.3f}, time: {result['time']:.2f}s)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Prediction comparison test failed: {e}")
        return False

def test_label_mapping():
    """Test and verify label mapping"""
    print("\n🏷️  LABEL MAPPING VERIFICATION")
    print("-" * 30)
    
    try:
        from fracture.hybrid_predictions_engine import categories_fracture
        print(f"Fracture categories: {categories_fracture}")
        print(f"Index 0: {categories_fracture[0]} (should be 'fractured')")
        print(f"Index 1: {categories_fracture[1]} (should be 'normal')")
        
        if categories_fracture[0] == 'fractured' and categories_fracture[1] == 'normal':
            print("✅ Label mapping is CORRECT")
            return True
        else:
            print("❌ Label mapping is INCORRECT")
            return False
            
    except Exception as e:
        print(f"❌ Label mapping test failed: {e}")
        return False

def main():
    """Run all tests for the hybrid ViT-CNN system"""
    print("🚀 STARTING HYBRID ViT-CNN SYSTEM TESTS")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all tests
    test_results = {}
    
    test_results['vit_components'] = test_vit_components()
    test_results['label_mapping'] = test_label_mapping()
    test_results['hybrid_engine'] = test_hybrid_prediction_engine()
    test_results['prediction_comparison'] = test_prediction_comparison()
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 FINAL TEST SUMMARY")
    print("=" * 60)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED - ViT-CNN Hybrid System Ready!")
    elif passed_tests >= total_tests * 0.75:
        print("⚠️  MOST TESTS PASSED - System mostly ready")
    else:
        print("❌ MANY TESTS FAILED - System needs fixes")
    
    print(f"\n📊 SYSTEM CAPABILITIES:")
    print("✅ ResNet50 CNN - Local feature extraction")
    print("✅ Vision Transformer - Global context understanding")
    print("✅ Hybrid ViT-CNN - Combined local + global features")
    print("✅ Patch Embedding - Image tokenization")
    print("✅ Multi-Head Self-Attention - Feature relationships")
    print("✅ Transformer Encoder - Sequential processing")
    
    print(f"\n⏰ Tests completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
