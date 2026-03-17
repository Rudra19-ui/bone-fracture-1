import React, { useState } from 'react';
import './MedicalAssistant.css';

const MedicalAssistant = () => {
  const [pdfFile, setPdfFile] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('upload');

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file && file.type === 'application/pdf') {
      setPdfFile(file);
      setError(null);
      setAnalysisResult(null);
    } else {
      setError('Please upload a valid PDF file');
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type === 'application/pdf') {
      setPdfFile(file);
      setError(null);
      setAnalysisResult(null);
    } else {
      setError('Please upload a valid PDF file');
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const analyzePDF = async () => {
    if (!pdfFile) {
      setError('Please upload a PDF file first');
      return;
    }

    setIsAnalyzing(true);
    setError(null);

    const formData = new FormData();
    formData.append('pdf_file', pdfFile);

    try {
      const response = await fetch('/api/medical-analysis/', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        setAnalysisResult(result);
        setActiveTab('results');
      } else {
        const errorData = await response.json();
        setError(errorData.error || 'Analysis failed');
      }
    } catch (err) {
      setError('Network error. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const renderSeverityBadge = (severity) => {
    const colors = {
      'NORMAL': 'success',
      'MODERATE': 'warning',
      'HIGH': 'danger'
    };
    return <span className={`severity-badge ${colors[severity] || 'info'}`}>{severity}</span>;
  };

  const renderResults = () => {
    if (!analysisResult) return null;

    const {
      report_summary,
      interpretation,
      severity_insight,
      recommended_actions,
      recovery_expectation,
      additional_advice,
      disclaimer,
      extracted_data
    } = analysisResult;

    return (
      <div className="medical-results">
        {/* Report Summary */}
        <div className="result-section">
          <h3>📋 Report Summary</h3>
          <div className="summary-grid">
            <div className="summary-item">
              <label>Fracture Detected:</label>
              <span className={extracted_data.fracture_detected === 'YES' ? 'fracture' : 'normal'}>
                {extracted_data.fracture_detected}
              </span>
            </div>
            <div className="summary-item">
              <label>Bone Type:</label>
              <span>{extracted_data.bone_type}</span>
            </div>
            <div className="summary-item">
              <label>Confidence:</label>
              <span>{extracted_data.confidence}%</span>
            </div>
            {extracted_data.location && (
              <div className="summary-item">
                <label>Location:</label>
                <span>{extracted_data.location}</span>
              </div>
            )}
          </div>
        </div>

        {/* Interpretation */}
        <div className="result-section">
          <h3>🧠 Interpretation</h3>
          <p>{interpretation}</p>
        </div>

        {/* Severity Insight */}
        <div className="result-section">
          <h3>⚠️ Severity Insight</h3>
          <p>{severity_insight}</p>
        </div>

        {/* Recommended Actions */}
        <div className="result-section">
          <h3>🏥 Recommended Actions</h3>
          <ul className="action-list">
            {recommended_actions.map((action, index) => (
              <li key={index} className="action-item">
                <span className="action-number">{index + 1}</span>
                {action}
              </li>
            ))}
          </ul>
        </div>

        {/* Recovery Expectation */}
        <div className="result-section">
          <h3>⏳ Recovery Expectation</h3>
          <p>{recovery_expectation}</p>
        </div>

        {/* Additional Advice */}
        <div className="result-section">
          <h3>💡 Additional Advice</h3>
          <ul className="advice-list">
            {additional_advice.map((advice, index) => (
              <li key={index}>{advice}</li>
            ))}
          </ul>
        </div>

        {/* Disclaimer */}
        <div className="result-section disclaimer">
          <h3>⚠️ Disclaimer</h3>
          <p>{disclaimer}</p>
        </div>
      </div>
    );
  };

  return (
    <div className="medical-assistant">
      <div className="medical-header">
        <h2>🏥 AI Medical Assistant</h2>
        <p>Analyze your bone fracture reports with AI-powered medical guidance</p>
      </div>

      <div className="medical-tabs">
        <button
          className={`tab-button ${activeTab === 'upload' ? 'active' : ''}`}
          onClick={() => setActiveTab('upload')}
        >
          📤 Upload Report
        </button>
        <button
          className={`tab-button ${activeTab === 'results' ? 'active' : ''}`}
          onClick={() => setActiveTab('results')}
          disabled={!analysisResult}
        >
          📊 Analysis Results
        </button>
      </div>

      <div className="tab-content">
        {activeTab === 'upload' && (
          <div className="upload-section">
            <div
              className="upload-area"
              onDrop={handleDrop}
              onDragOver={handleDragOver}
            >
              {pdfFile ? (
                <div className="file-preview">
                  <div className="file-icon">📄</div>
                  <div className="file-info">
                    <p className="file-name">{pdfFile.name}</p>
                    <p className="file-size">{(pdfFile.size / 1024 / 1024).toFixed(2)} MB</p>
                  </div>
                  <button
                    className="remove-file"
                    onClick={() => {
                      setPdfFile(null);
                      setAnalysisResult(null);
                      setError(null);
                    }}
                  >
                    ✕
                  </button>
                </div>
              ) : (
                <div className="upload-placeholder">
                  <div className="upload-icon">📁</div>
                  <p>Drag & drop your PDF report here</p>
                  <p>or click to browse</p>
                  <input
                    type="file"
                    accept=".pdf"
                    onChange={handleFileUpload}
                    style={{ display: 'none' }}
                  />
                </div>
              )}
            </div>

            {error && (
              <div className="error-message">
                <span className="error-icon">⚠️</span>
                {error}
              </div>
            )}

            <button
              className="analyze-button"
              onClick={analyzePDF}
              disabled={!pdfFile || isAnalyzing}
            >
              {isAnalyzing ? (
                <div className="loading-content">
                  <div className="spinner"></div>
                  Analyzing report...
                </div>
              ) : (
                '🔬 Analyze Report'
              )}
            </button>
          </div>
        )}

        {activeTab === 'results' && renderResults()}
      </div>
    </div>
  );
};

export default MedicalAssistant;
