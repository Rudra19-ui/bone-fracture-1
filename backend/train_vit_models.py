#!/usr/bin/env python3
"""
Training Script for Vision Transformer (ViT) Models
Trains both pure ViT and hybrid ViT-CNN models for bone fracture detection
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Add fracture module to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from fracture.vit_model import create_vit_model, create_hybrid_vit_cnn_model
    print("✅ Successfully imported ViT models")
except ImportError as e:
    print(f"❌ Failed to import ViT models: {e}")
    sys.exit(1)

def load_path(image_dir, part):
    """Load dataset from local directory"""
    dataset = []
    labels_map = {'study1_positive': 'fractured', 'study1_negative': 'normal'}
    
    if not os.path.exists(image_dir):
        print(f"❌ Directory not found: {image_dir}")
        return dataset
    
    part_dir = os.path.join(image_dir, part)
    if not os.path.exists(part_dir):
        print(f"❌ Part directory not found: {part_dir}")
        return dataset
    
    for patient_folder in os.listdir(part_dir):
        patient_path = os.path.join(part_dir, patient_folder)
        if not os.path.isdir(patient_path):
            continue
            
        for study_folder in os.listdir(patient_path):
            study_path = os.path.join(patient_path, study_folder)
            if not os.path.isdir(study_path):
                continue
                
            study_type = study_folder.split('_')[1] if '_' in study_folder else study_folder
            label = labels_map.get(study_type, 'unknown')
            
            if label == 'unknown':
                continue
                
            for img_file in os.listdir(study_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(study_path, img_file)
                    if os.path.exists(img_path):
                        dataset.append({
                            'body_part': part,
                            'patient_id': patient_folder,
                            'study_type': study_folder,
                            'label': label,
                            'image_path': img_path
                        })
    
    print(f"📊 Loaded {len(dataset)} images for {part}")
    return dataset

def load_kaggle_data(part):
    """Load external Kaggle dataset"""
    try:
        import kagglehub
        path = kagglehub.dataset_download("pkdarabi/bone-fracture-detection-computer-vision-project")
        print(f"📥 Kaggle data found at: {path}")
        
        kaggle_dataset = []
        class_to_part = {
            0: "Elbow", 1: "Hand", 2: "Wrist", 
            3: "Shoulder", 4: "Shoulder", 5: "Shoulder", 6: "Wrist"
        }
        
        for split in ['train', 'test']:
            img_dir = os.path.join(path, "BoneFractureYolo8", split, "images")
            label_dir = os.path.join(path, "BoneFractureYolo8", split, "labels")
            
            if not os.path.exists(img_dir): 
                continue
            
            for img_name in os.listdir(img_dir):
                img_path = os.path.join(img_dir, img_name)
                label_name = os.path.splitext(img_name)[0] + ".txt"
                label_path = os.path.join(label_dir, label_name)
                
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            class_idx = int(lines[0].split()[0])
                            detected_part = class_to_part.get(class_idx, "Unknown")
                            
                            if detected_part == part:
                                # Map YOLO classes to fracture/normal
                                # Assuming class 0,1,2,5,6 are fractures, 3,4 are normal
                                is_fracture = class_idx in [0, 1, 2, 5, 6]
                                label = 'fractured' if is_fracture else 'normal'
                                
                                kaggle_dataset.append({
                                    'body_part': part,
                                    'patient_id': f"kaggle_{split}_{class_idx}",
                                    'study_type': f"kaggle_{split}",
                                    'label': label,
                                    'image_path': img_path
                                })
        
        print(f"📊 Loaded {len(kaggle_dataset)} Kaggle images for {part}")
        return kaggle_dataset
        
    except Exception as e:
        print(f"❌ Failed to load Kaggle data: {e}")
        return []

def train_vit_model(part, model_type="hybrid"):
    """Train ViT-based model for specific body part"""
    print(f"\n🚀 Starting ViT training for {part} - {model_type} model")
    
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    image_dir = THIS_FOLDER + '/Dataset/'
    
    # Load datasets
    print("📥 Loading datasets...")
    local_data = load_path(image_dir, part)
    kaggle_data = load_kaggle_data(part)
    
    # Combine datasets
    data = local_data + kaggle_data
    print(f"📊 Total dataset size: {len(data)} images")
    
    if len(data) == 0:
        print(f"❌ No data found for {part}")
        return
    
    # Create DataFrame
    labels = []
    filepaths = []
    
    for row in data:
        labels.append(row['label'])
        filepaths.append(row['image_path'])
    
    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')
    images = pd.concat([filepaths, labels], axis=1)
    
    # Print class distribution
    print(f"📊 Class distribution:")
    print(images['Label'].value_counts())
    
    # Split dataset
    train_df, test_df = train_test_split(images, train_size=0.8, shuffle=True, random_state=42, stratify=labels)
    
    # Data generators
    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
        validation_split=0.2,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input
    )
    
    # Create data flows
    train_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=16,  # Smaller batch size for ViT
        shuffle=True,
        seed=42,
        subset='training'
    )
    
    val_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=16,
        shuffle=True,
        seed=42,
        subset='validation'
    )
    
    test_images = test_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=16,
        shuffle=False
    )
    
    # Create model
    print(f"🏗️  Creating {model_type} ViT model...")
    
    if model_type == "hybrid":
        model = create_hybrid_vit_cnn_model(input_shape=(224, 224, 3), num_classes=2)
        print("✅ Hybrid ViT-CNN model created")
    elif model_type == "pure":
        model = create_vit_model(input_shape=(224, 224, 3), num_classes=2, model_size='base')
        print("✅ Pure ViT model created")
    else:
        print("❌ Unknown model type")
        return
    
    # Calculate class weights
    unique_classes = np.unique(train_df['Label'])
    weights = class_weight.compute_class_weight('balanced', classes=unique_classes, y=train_df['Label'])
    class_weights_dict = dict(enumerate(weights))
    print(f"📊 Class weights: {class_weights_dict}")
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=1e-4),  # Lower learning rate for ViT
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall(name='recall')]
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
        ModelCheckpoint(
            filepath=f'weights/ViT_{model_type}_{part}_frac.h5',
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True
        )
    ]
    
    # Train model
    print(f"🎯 Training {model_type} ViT model for {part}...")
    
    history = model.fit(
        train_images,
        validation_data=val_images,
        epochs=30,  # More epochs for ViT
        callbacks=callbacks,
        class_weight=class_weights_dict,
        verbose=1
    )
    
    # Evaluate model
    print(f"📊 Evaluating {model_type} ViT model...")
    results = model.evaluate(test_images, verbose=0)
    
    print(f"🎯 {part} {model_type} ViT Results:")
    print(f"   Loss: {results[0]:.4f}")
    print(f"   Accuracy: {results[1]:.4f} ({results[1]*100:.2f}%)")
    print(f"   Recall: {results[2]:.4f} ({results[2]*100:.2f}%)")
    
    # Save final model
    weights_dir = os.path.join(THIS_FOLDER, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    
    final_model_path = os.path.join(weights_dir, f'ViT_{model_type}_{part}_frac_final.h5')
    model.save_weights(final_model_path)
    print(f"💾 Saved final weights to: {final_model_path}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{part} {model_type} ViT - Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{part} {model_type} ViT - Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plot_path = os.path.join(weights_dir, f'ViT_{model_type}_{part}_training_history.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"📊 Saved training plot to: {plot_path}")
    
    return model, history, results

def train_all_vit_models():
    """Train ViT models for all body parts"""
    print("🚀 Starting comprehensive ViT training for all body parts")
    
    body_parts = ['Elbow', 'Hand', 'Shoulder']
    model_types = ['hybrid', 'pure']
    
    results = {}
    
    for part in body_parts:
        print(f"\n{'='*60}")
        print(f"🦴 Training for {part}")
        print(f"{'='*60}")
        
        for model_type in model_types:
            try:
                model, history, eval_results = train_vit_model(part, model_type)
                results[f"{part}_{model_type}"] = eval_results
                
                # Clear memory
                tf.keras.backend.clear_session()
                
            except Exception as e:
                print(f"❌ Failed to train {model_type} ViT for {part}: {e}")
                results[f"{part}_{model_type}"] = None
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 TRAINING SUMMARY")
    print(f"{'='*60}")
    
    for key, result in results.items():
        if result:
            print(f"✅ {key}: Accuracy={result[1]:.3f}, Recall={result[2]:.3f}")
        else:
            print(f"❌ {key}: Training failed")
    
    return results

if __name__ == "__main__":
    print("🚀 Vision Transformer Training Script")
    print("=" * 50)
    
    # Check for weights directory
    weights_dir = os.path.join(os.path.dirname(__file__), 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    
    # Train models
    try:
        results = train_all_vit_models()
        print("\n🎉 ViT training completed!")
        
        # Save results summary
        import json
        results_file = os.path.join(weights_dir, 'vit_training_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to: {results_file}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
    
    print("🏁 Script finished")
