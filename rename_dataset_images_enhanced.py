#!/usr/bin/env python3
"""
Bone Fracture Dataset Image Renamer - Enhanced Version
Renames image files with bone type and study type suffixes
Features: Dry-run mode, progress tracking, conflict detection
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import argparse

# Configure logging
def setup_logging(log_file=None, verbose=False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)

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

class DatasetRenamer:
    """Main class for dataset image renaming"""
    
    def __init__(self, dry_run=False, verbose=False):
        self.dry_run = dry_run
        self.verbose = verbose
        self.logger = setup_logging(verbose=verbose)
        self.stats = {
            'processed': 0,
            'renamed': 0,
            'skipped': 0,
            'errors': 0
        }
    
    def get_bone_code(self, folder_path):
        """Extract bone type code from folder path"""
        path_lower = str(folder_path).lower()
        for bone_type, code in BONE_TYPE_MAP.items():
            if bone_type in path_lower:
                return code
        return None
    
    def get_study_code(self, folder_path):
        """Extract study type code from folder path"""
        path_lower = str(folder_path).lower()
        for study_type, code in STUDY_TYPE_MAP.items():
            if study_type in path_lower:
                return code
        return None
    
    def already_has_suffix(self, filename):
        """Check if file already has our suffixes"""
        suffixes = ['.1', '.2', '.3', '.5', '.6']
        return any(suffix in filename for suffix in suffixes)
    
    def check_conflict(self, file_path, new_name):
        """Check if new filename would conflict with existing file"""
        new_path = file_path.parent / new_name
        return new_path.exists() and new_path != file_path
    
    def rename_image_file(self, file_path):
        """Rename a single image file with appropriate suffixes"""
        try:
            file_path = Path(file_path)
            self.stats['processed'] += 1
            
            # Skip if not an image file
            if file_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
                if self.verbose:
                    self.logger.debug(f"SKIP: {file_path.name} (not an image)")
                self.stats['skipped'] += 1
                return False
            
            # Skip if already has suffix
            if self.already_has_suffix(file_path.name):
                self.logger.info(f"SKIP: {file_path.name} (already has suffix)")
                self.stats['skipped'] += 1
                return False
            
            # Get bone and study codes
            bone_code = self.get_bone_code(file_path)
            study_code = self.get_study_code(file_path)
            
            if not bone_code or not study_code:
                self.logger.warning(f"SKIP: {file_path.name} (missing bone or study code)")
                self.logger.warning(f"  Bone code: {bone_code}, Study code: {study_code}")
                self.stats['skipped'] += 1
                return False
            
            # Create new filename
            old_name = file_path.name
            stem = file_path.stem
            extension = file_path.suffix
            new_name = f"{stem}{bone_code}{study_code}{extension}"
            new_path = file_path.parent / new_name
            
            # Check for conflicts
            if self.check_conflict(file_path, new_name):
                self.logger.error(f"CONFLICT: {new_name} already exists")
                self.stats['errors'] += 1
                return False
            
            # Rename file (or simulate in dry-run mode)
            if self.dry_run:
                self.logger.info(f"DRY-RUN: {old_name} → {new_name}")
                self.stats['renamed'] += 1
            else:
                file_path.rename(new_path)
                self.logger.info(f"OLD: {old_name}")
                self.logger.info(f"NEW: {new_name}")
                self.stats['renamed'] += 1
            
            return True
            
        except Exception as e:
            self.logger.error(f"ERROR renaming {file_path}: {e}")
            self.stats['errors'] += 1
            return False
    
    def process_dataset(self, root_folder):
        """Process entire dataset recursively"""
        root_path = Path(root_folder)
        
        if not root_path.exists():
            self.logger.error(f"Root folder does not exist: {root_folder}")
            return
        
        self.logger.info(f"Processing dataset: {root_folder}")
        initial_stats = self.stats.copy()
        
        # Walk through all folders
        for folder_path in root_path.rglob('*'):
            if folder_path.is_file():
                self.rename_image_file(folder_path)
        
        # Log summary for this folder
        renamed = self.stats['renamed'] - initial_stats['renamed']
        self.logger.info(f"Folder {root_folder}: {renamed} files renamed")
    
    def find_dataset_root(self):
        """Find the Dataset folder automatically"""
        current_dir = Path.cwd()
        
        # Search paths to check
        search_paths = [
            current_dir,
            current_dir.parent,
            current_dir.parent.parent,
            current_dir / 'Bone-Fracture-Detection-master',
            current_dir.parent / 'Bone-Fracture-Detection-master'
        ]
        
        for search_path in search_paths:
            potential_root = search_path / 'Dataset'
            if potential_root.exists():
                self.logger.info(f"Dataset root found: {potential_root}")
                return potential_root
        
        self.logger.error("Dataset folder not found!")
        self.logger.info("Please run this script from the project directory")
        return None
    
    def print_summary(self):
        """Print processing summary"""
        self.logger.info("=" * 50)
        self.logger.info("PROCESSING SUMMARY")
        self.logger.info("=" * 50)
        self.logger.info(f"Total files processed: {self.stats['processed']}")
        self.logger.info(f"Files renamed: {self.stats['renamed']}")
        self.logger.info(f"Files skipped: {self.stats['skipped']}")
        self.logger.info(f"Errors encountered: {self.stats['errors']}")
        
        if self.dry_run:
            self.logger.info(f"Mode: DRY-RUN (no actual changes made)")
        
        self.logger.info("=" * 50)
    
    def run(self, dataset_folders=None):
        """Main execution method"""
        if dataset_folders is None:
            dataset_folders = ['train_valid', 'test', 'test_valid']
        
        # Find dataset root
        dataset_root = self.find_dataset_root()
        if not dataset_root:
            return False
        
        # Process each dataset folder
        for folder_name in dataset_folders:
            folder_path = dataset_root / folder_name
            if folder_path.exists():
                self.process_dataset(folder_path)
            else:
                self.logger.warning(f"Folder not found: {folder_path}")
        
        # Print summary
        self.print_summary()
        
        return self.stats['errors'] == 0

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(
        description='Rename bone fracture dataset images with bone and study type suffixes'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be renamed without actually doing it'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--log-file',
        help='Log file path (default: rename_log.txt)'
    )
    parser.add_argument(
        '--folders',
        nargs='+',
        default=['train_valid', 'test', 'test_valid'],
        help='Dataset folders to process'
    )
    
    args = parser.parse_args()
    
    # Setup logging with file if specified
    if args.log_file:
        logger = setup_logging(args.log_file, args.verbose)
    
    # Create renamer instance
    renamer = DatasetRenamer(dry_run=args.dry_run, verbose=args.verbose)
    
    # Run the renaming process
    success = renamer.run(args.folders)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
