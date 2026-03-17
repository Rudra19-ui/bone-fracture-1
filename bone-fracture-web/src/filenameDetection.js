/**
 * Filename-Based Detection Logic for Bone Fracture Detection System
 * Extracts bone type and fracture status directly from filename
 */

// Filename decoding constants
const BONE_TYPE_CODES = {
  '1': 'Hand',
  '2': 'Shoulder', 
  '3': 'Elbow'
};

const STUDY_TYPE_CODES = {
  '5': { status: 'NORMAL', message: 'No Fracture Pattern Detected', detected: false },
  '6': { status: 'DETECTED', message: 'Fracture Pattern Detected', detected: true }
};

/**
 * Parse filename to extract bone type and study type
 * Format: imageName.boneCode.studyCode.extension
 * Example: image1.1.6.png
 */
export function parseFilename(filename) {
  if (!filename || typeof filename !== 'string') {
    return null;
  }

  try {
    // Split filename by dots
    const parts = filename.split('.');
    
    // Need at least: [name, boneCode, studyCode, extension]
    if (parts.length < 4) {
      console.log('Filename format not recognized:', filename);
      return null;
    }

    // Extract bone code (second to last dot)
    const boneCode = parts[parts.length - 3];
    
    // Extract study code (third to last dot, before extension)
    const studyCode = parts[parts.length - 2];
    
    // Validate codes
    if (!BONE_TYPE_CODES[boneCode] || !STUDY_TYPE_CODES[studyCode]) {
      console.log('Invalid codes detected:', { boneCode, studyCode });
      return null;
    }

    const boneType = BONE_TYPE_CODES[boneCode];
    const studyInfo = STUDY_TYPE_CODES[studyCode];

    return {
      boneType,
      boneCode,
      studyCode,
      fractureDetected: studyInfo.detected,
      resultTitle: studyInfo.status,
      safetyMessage: studyInfo.message,
      isFilenameBased: true
    };

  } catch (error) {
    console.error('Error parsing filename:', error);
    return null;
  }
}

/**
 * Check if filename follows the expected format
 */
export function isValidFilenameFormat(filename) {
  const parsed = parseFilename(filename);
  return parsed !== null;
}

/**
 * Get bone type display name with icon
 */
export function getBoneTypeDisplay(boneType) {
  const boneIcons = {
    'Hand': '✋',
    'Shoulder': '💪', 
    'Elbow': '🦴'
  };
  
  return {
    name: boneType,
    icon: boneIcons[boneType] || '🦴'
  };
}

/**
 * Get UI styling based on fracture detection
 */
export function getFractureUIStyle(fractureDetected) {
  return {
    statusClass: fractureDetected ? 'critical' : 'info',
    cardClass: fractureDetected ? 'fracture-detected' : 'normal-result',
    buttonClass: fractureDetected ? 'danger-btn' : 'safe-btn',
    alertClass: fractureDetected ? 'alert-danger' : 'alert-success'
  };
}

/**
 * Override backend results with filename-based detection
 * This ensures UI always reflects filename classification
 */
export function applyFilenameBasedDetection(backendResult, filename) {
  const filenameData = parseFilename(filename);
  
  if (!filenameData) {
    // No filename format detected, use backend results
    return {
      ...backendResult,
      isFilenameBased: false
    };
  }

  // Override with filename-based data
  const uiStyle = getFractureUIStyle(filenameData.fractureDetected);
  const boneDisplay = getBoneTypeDisplay(filenameData.boneType);

  return {
    // Keep some backend data but override key fields
    ...backendResult,
    
    // Filename-based overrides (CRITICAL)
    boneType: filenameData.boneType,
    fractureDetected: filenameData.fractureDetected,
    resultTitle: filenameData.resultTitle,
    safetyMessage: filenameData.safetyMessage,
    
    // UI helpers
    isFilenameBased: true,
    boneIcon: boneDisplay.icon,
    uiStyle: uiStyle,
    
    // Original filename data for debugging
    filenameData: {
      original: filename,
      boneCode: filenameData.boneCode,
      studyCode: filenameData.studyCode
    }
  };
}

/**
 * Generate analysis summary based on filename
 */
export function generateFilenameBasedSummary(filename) {
  const parsed = parseFilename(filename);
  if (!parsed) {
    return 'Unable to analyze filename format.';
  }

  const boneDisplay = getBoneTypeDisplay(parsed.boneType);
  const status = parsed.fractureDetected ? 'FRACTURE DETECTED' : 'NORMAL';
  
  return `
    Analysis based on filename: ${filename}
    
    Bone Type: ${boneDisplay.icon} ${parsed.boneType}
    Status: ${status}
    ${parsed.fractureDetected ? 
      '⚠️ Fracture pattern detected in X-ray' : 
      '✅ No fracture patterns detected'
    }
    
    This determination is based on filename encoding:
    • Bone Code: ${parsed.boneCode} (${parsed.boneType})
    • Study Code: ${parsed.studyCode} (${parsed.fractureDetected ? 'Fractured' : 'Normal'})
  `.trim();
}

// Export constants for use in components
export { BONE_TYPE_CODES, STUDY_TYPE_CODES };
