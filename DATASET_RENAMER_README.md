# Bone Fracture Dataset Image Renamer

## 🎯 Purpose

This script renames all image files in the bone fracture dataset with specific numeric suffixes that encode bone type and study type information. This enables the frontend to identify bone types and fracture status directly from filenames.

## 📊 Naming Convention

### **Format**: `originalfilename.boneCode.studyCode.extension`

### **Bone Type Codes**:
- `.1` → Hand
- `.2` → Shoulder  
- `.3` → Elbow

### **Study Type Codes**:
- `.6` → Positive (fractured)
- `.5` → Negative (normal)

### **Examples**:
```
image1.png          → image1.1.6.png    (Hand + Fractured)
image2.png          → image2.1.5.png    (Hand + Normal)
image3.png          → image3.2.6.png    (Shoulder + Fractured)
image4.png          → image4.3.5.png    (Elbow + Normal)
```

## 🚀 Usage

### **Simple Usage**:
```bash
python run_rename.py
```

### **Advanced Usage**:
```bash
# Dry run (preview changes without executing)
python rename_dataset_images_enhanced.py --dry-run

# Verbose output
python rename_dataset_images_enhanced.py --verbose

# Save log to file
python rename_dataset_images_enhanced.py --log-file rename_log.txt

# Process specific folders
python rename_dataset_images_enhanced.py --folders train_valid test
```

## 📁 Dataset Structure

The script processes this structure:
```
Dataset/
├── train_valid/
│   ├── Elbow/
│   │   ├── patientXXXX/
│   │   │   └── study1_positive/
│   │   │       ├── image1.png → image1.3.6.png
│   │   │       └── image2.png → image2.3.6.png
│   │   └── patientYYYY/
│   │       └── study1_negative/
│   │           ├── image1.png → image1.3.5.png
│   └── Hand/
│       └── patientZZZZ/
│           └── study1_positive/
│               ├── image1.png → image1.1.6.png
└── test/
    ├── Elbow/
    └── Hand/
```

## ✅ Features

### **Core Features**:
- ✅ **Recursive Processing**: Traverses all subfolders
- ✅ **Smart Detection**: Automatically identifies bone and study types
- ✅ **Idempotent**: Skips files that already have suffixes
- ✅ **Conflict Detection**: Prevents filename conflicts
- ✅ **Comprehensive Logging**: Tracks all changes

### **Safety Features**:
- ✅ **Dry Run Mode**: Preview changes before executing
- ✅ **Error Handling**: Continues processing if individual files fail
- ✅ **Backup Logging**: Saves detailed log files
- ✅ **Progress Tracking**: Shows processing statistics

### **Advanced Features**:
- ✅ **Command Line Interface**: Flexible options
- ✅ **Verbose Mode**: Detailed debugging output
- ✅ **Custom Folders**: Process specific dataset subsets
- ✅ **Statistics Summary**: Complete processing report

## 📊 Test Results

### **Dry Run Summary**:
```
===============================================
PROCESSING SUMMARY
===============================================
Total files processed: 20,335
Files renamed: 19,530
Files skipped: 805
Errors encountered: 0
Mode: DRY-RUN (no actual changes made)
===============================================
```

### **Processing Breakdown**:
- **train_valid**: ~18,144 files processed
- **test**: ~1,386 files processed
- **test_valid**: Not found (optional folder)

## 🔧 Technical Details

### **File Detection**:
- **Supported Formats**: `.png`, `.jpg`, `.jpeg`
- **Bone Type Detection**: Case-insensitive folder matching
- **Study Type Detection**: Study folder analysis
- **Suffix Detection**: Checks for existing `.1`, `.2`, `.3`, `.5`, `.6`

### **Error Handling**:
- **Missing Folders**: Graceful skipping with warnings
- **Permission Issues**: Error logging and continuation
- **Invalid Paths**: Safe path resolution
- **File Conflicts**: Pre-rename conflict detection

### **Logging**:
- **Console Output**: Real-time progress updates
- **File Logging**: Detailed operation logs
- **Statistics**: Comprehensive summary reports
- **Error Tracking**: Individual error details

## 🛠️ Installation & Setup

### **Requirements**:
- Python 3.6+
- Standard library only (no external dependencies)

### **Setup**:
1. Place scripts in `Bone-Fracture-Detection-master/` directory
2. Ensure `Dataset/` folder exists in the same directory
3. Run the script

## 📝 File Descriptions

### **`rename_dataset_images_enhanced.py`**
- Main renaming script with full features
- Command-line interface
- Advanced error handling and logging

### **`run_rename.py`**
- Simple wrapper script
- One-click execution
- Basic error checking

### **`rename_log.txt`** (generated)
- Detailed operation log
- All file changes recorded
- Error tracking

## 🎯 Frontend Integration

After renaming, the frontend can decode filenames:

```javascript
function decodeFilename(filename) {
    const parts = filename.split('.');
    const boneCode = parts[parts.length - 2];
    const studyCode = parts[parts.length - 1].split('.')[0];
    
    const boneTypes = { '1': 'Hand', '2': 'Shoulder', '3': 'Elbow' };
    const studyTypes = { '5': 'Normal', '6': 'Fractured' };
    
    return {
        boneType: boneTypes[boneCode],
        studyType: studyTypes[studyCode]
    };
}
```

## ⚠️ Important Notes

### **Before Running**:
1. **Backup**: Create a backup of your dataset
2. **Dry Run**: Always test with `--dry-run` first
3. **Permissions**: Ensure write access to dataset folders

### **After Running**:
1. **Verify**: Check the log file for any errors
2. **Test**: Verify frontend can decode filenames correctly
3. **Backup**: Save the log file for reference

### **Safety**:
- ✅ Script is idempotent (safe to run multiple times)
- ✅ Preserves original filenames (only appends suffixes)
- ✅ Maintains folder structure
- ✅ Handles edge cases gracefully

## 🎉 Success Criteria

The script successfully processes your dataset when:
- ✅ All image files have appropriate suffixes
- ✅ No filename conflicts occur
- ✅ Folder structure is preserved
- ✅ Log file shows successful completion
- ✅ Frontend can decode the new filenames

## 📞 Support

For issues or questions:
1. Check the log file for error details
2. Run with `--verbose` for debugging
3. Use `--dry-run` to test changes
4. Verify dataset structure matches expected format
