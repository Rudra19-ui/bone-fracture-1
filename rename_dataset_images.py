#!/usr/bin/env python3
"""
Bone Fracture Dataset Image Renamer
Renames image files with bone type and study type suffixes
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('image_rename_log.txt'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Bone type mapping
BONE_TYPE_MAP = {
    'hand': '.1',
    'shoulder': '.2', 
    'elbow': '.3'
}

# Study type mapping
STUDY_TYPE_MAP = {
    'study1_positive': '.6',
    'study1_negative': '.5'
}

def get_bone_code(folder_path):
    """Extract bone type code from folder path"""
    path_lower = str(folder_path).lower()
    for bone_type, code in BONE_TYPE_MAP.items():
        if bone_type in path_lower:
            return code
    return None

def get_study_code(folder_path):
    """Extract study type code from folder path"""
    path_lower = str(folder_path).lower()
    for study_type, code in STUDY_TYPE_MAP.items():
        if study_type in path_lower:
            return code
    return None

def already_has_suffix(filename):
    """Check if file already has our suffixes"""
    suffixes = ['.1', '.2', '.3', '.5', '.6']
    return any(suffix in filename for suffix in suffixes)

def rename_image_file(file_path):
    """Rename a single image file with appropriate suffixes"""
    try:
        file_path = Path(file_path)
        
        # Skip if not an image file
        if file_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
            return False
        
        # Skip if already has suffix
        if already_has_suffix(file_path.name):
            logger.info(f"SKIP: {file_path.name} (already has suffix)")
            return False
        
        # Get bone and study codes
        bone_code = get_bone_code(file_path)
        study_code = get_study_code(file_path)
        
        if not bone_code or not study_code:
            logger.warning(f"SKIP: {file_path.name} (missing bone or study code)")
            return False
        
        # Create new filename
        old_name = file_path.name
        stem = file_path.stem
        extension = file_path.suffix
        new_name = f"{stem}{bone_code}{study_code}{extension}"
        new_path = file_path.parent / new_name
        
        # Rename file
        file_path.rename(new_path)
        logger.info(f"OLD: {old_name}")
        logger.info(f"NEW: {new_name}")
        return True
        
    except Exception as e:
        logger.error(f"ERROR renaming {file_path}: {e}")
        return False

def process_dataset(root_folder):
    """Process entire dataset recursively"""
    root_path = Path(root_folder)
    
    if not root_path.exists():
        logger.error(f"Root folder does not exist: {root_folder}")
        return
    
    logger.info(f"Processing dataset: {root_folder}")
    renamed_count = 0
    
    # Walk through all folders
    for folder_path in root_path.rglob('*'):
        if folder_path.is_file():
            if rename_image_file(folder_path):
                renamed_count += 1
    
    logger.info(f"Total files renamed: {renamed_count}")

def main():
    """Main function"""
    # Dataset folders to process
    dataset_folders = [
        'train_valid',
        'test', 
        'test_valid'
    ]
    
    # Find dataset root
    current_dir = Path.cwd()
    dataset_root = None
    
    # Look for Dataset folder
    for parent in [current_dir, current_dir.parent, current_dir.parent.parent]:
        potential_root = parent / 'Dataset'
        if potential_root.exists():
            dataset_root = potential_root
            break
    
    if not dataset_root:
        logger.error("Dataset folder not found!")
        logger.info("Please run this script from the project directory")
        return
    
    logger.info(f"Dataset root found: {dataset_root}")
    
    # Process each dataset folder
    for folder_name in dataset_folders:
        folder_path = dataset_root / folder_name
        if folder_path.exists():
            process_dataset(folder_path)
        else:
            logger.warning(f"Folder not found: {folder_path}")
    
    logger.info("Renaming completed!")

if __name__ == "__main__":
    main()
