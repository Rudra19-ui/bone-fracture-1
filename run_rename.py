#!/usr/bin/env python3
"""
Simple wrapper script for dataset renaming
"""

import os
import sys
from pathlib import Path

def main():
    """Simple execution"""
    print("🦴 Bone Fracture Dataset Image Renamer")
    print("=" * 50)
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    dataset_path = current_dir / 'Dataset'
    
    if not dataset_path.exists():
        print("❌ Dataset folder not found!")
        print("Please run this script from the Bone-Fracture-Detection-master directory")
        return False
    
    print(f"✅ Dataset found: {dataset_path}")
    
    # Import and run the enhanced renamer
    try:
        from rename_dataset_images_enhanced import DatasetRenamer
        
        # Create renamer instance
        renamer = DatasetRenamer(dry_run=False, verbose=True)
        
        # Run the process
        success = renamer.run()
        
        if success:
            print("🎉 Renaming completed successfully!")
        else:
            print("⚠️  Renaming completed with some errors")
        
        return success
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
