import React, { useState, useEffect, useRef } from 'react';
import './Chatbot.css';

const Chatbot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    { id: 1, text: "Hello! I'm your FractureAI Medical Advisor. I can analyze your X-ray reports (Images or PDF), provide healing timelines, and suggest precautions. How can I assist you today?", sender: 'bot' }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [filePreview, setFilePreview] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [backendStatus, setBackendStatus] = useState('checking'); // 'online', 'offline', 'standalone'
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  const localKnowledgeBase = {
    "greetings": ["hi", "hello", "hey", "hola", "what are you doing"],
    "accuracy": ["accurate", "reliable", "precision", "accuracy"],
    "help": ["help", "what can you do", "guide", "analyze", "report"],
    "models": ["model", "vit", "resnet", "architecture", "gemini"],
    "bones": ["elbow", "hand", "shoulder", "wrist", "parts", "body"]
  };

  const getLocalResponse = (message) => {
    const msg = message.toLowerCase();
    
    if (msg.includes("what are you doing")) {
        return "I'm currently assisting as your digital Medical Advisor. I'm ready to help you understand your reports or discuss precautions for various bone fractures. How can I help you right now?";
    }
    
    if (localKnowledgeBase.greetings.some(k => msg.includes(k))) 
      return "Greetings. I am your FractureAI Advisor. I'm here to assist with your fracture analysis and answer general orthopedic health questions. What's on your mind?";
    
    if (msg.includes("wrist") || msg.includes("radius") || msg.includes("ulna"))
      return "For a suspected Wrist (Distal Radius/Ulna) fracture: 1. Immediate orthopedic consultation is required for alignment assessment. 2. Immobilize the joint using a wrist splint or cast. 3. Avoid lifting any objects. Typical healing for non-displaced wrist fractures is 6-8 weeks.";

    if (msg.includes("report") || msg.includes("analyze") || msg.includes("file") || msg.includes("pdf"))
      return "I see you're referencing a report. In Limited Mode, I can't perform live vision analysis. However, based on typical fracture reports: 1. Seek an orthopedic consultation. 2. Immobilize the joint. 3. Follow the 'Next Steps' in your report exactly. If you tell me the bone type, I can provide more specific healing info.";

    if (localKnowledgeBase.accuracy.some(k => msg.includes(k)))
      return "The system is highly precise, achieving 92.4% accuracy for fractures and 99.2% for anatomical classification. It leverages ResNet50 for vision and Gemini AI for logical analysis.";
    
    if (localKnowledgeBase.models.some(k => msg.includes(k)))
      return "Our architecture is an ensemble of ResNet50 (for image features) and Google Gemini AI (for medical reasoning). This combination ensures both speed and accuracy.";
    
    if (localKnowledgeBase.bones.some(k => msg.includes(k)))
      return "We currently support high-accuracy detection for Elbow, Hand, and Shoulder regions. We're also expanding support for wrists and other anatomical areas!";
    
    if (localKnowledgeBase.help.some(k => msg.includes(k)))
      return "To analyze a report, you can drag and drop your image or PDF into this chat window, or use the 📎 icon. Once uploaded, I'll provide a detailed analysis and precautions.";
    
    return "I'm currently operating in limited mode. While I can't access my full AI cloud right now, I can still answer basic questions about bone health and precautions. What specific bone or symptom would you like to discuss?";
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  useEffect(() => {
    const checkBackend = async () => {
      try {
        const response = await fetch(`${getApiBase()}/api/chatbot/`, {
          method: 'OPTIONS',
          timeout: 5000
        });
        if (response.ok) {
          setBackendStatus('online');
        } else {
          setBackendStatus('offline');
        }
      } catch (error) {
        setBackendStatus('offline');
      }
    };

    checkBackend();
    const interval = setInterval(checkBackend, 30000); // Check every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    validateAndSetFile(file);
  };

  const validateAndSetFile = (file) => {
    if (!file) return;
    
    const isImage = file.type.startsWith('image/');
    const isPDF = file.type === 'application/pdf';
    
    if (isImage || isPDF) {
      setSelectedFile(file);
      if (isImage) {
        const reader = new FileReader();
        reader.onload = (e) => setFilePreview(e.target.result);
        reader.readAsDataURL(file);
      } else {
        setFilePreview('PDF_DOCUMENT');
      }
    } else {
      alert("Please upload an image or a PDF report.");
    }
  };

  const onDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = () => {
    setIsDragging(false);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    validateAndSetFile(file);
  };

  const removeFile = () => {
    setSelectedFile(null);
    setFilePreview(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const getApiBase = () => {
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
      return 'http://localhost:8000';
    }
    return 'https://bone-fracture-backend-or69.onrender.com';
  };

  const handleSend = async () => {
    if (!inputValue.trim() && !selectedFile) return;

    const userMsgText = inputValue;
    const currentFile = selectedFile;
    const currentPreview = filePreview;
    
    const userMessage = { 
        id: Date.now(), 
        text: userMsgText || (currentFile?.type === 'application/pdf' ? "Attached a PDF report." : "Attached an image."), 
        sender: 'user',
        image: (currentFile && currentFile.type.startsWith('image/')) ? currentPreview : null,
        isPDF: currentFile?.type === 'application/pdf'
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setSelectedFile(null);
    setFilePreview(null);
    setIsTyping(true);

    try {
      const formData = new FormData();
      formData.append('message', userMsgText);
      formData.append('history', JSON.stringify(messages.map(m => ({ sender: m.sender, text: m.text })).slice(-8)));
      if (currentFile) formData.append('file', currentFile);

      const response = await fetch(`${getApiBase()}/api/chatbot/`, {
          method: 'POST',
          body: formData,
      });

      if (!response.ok) throw new Error('Backend error');
      const data = await response.json();

      const botMessage = { id: Date.now() + 1, text: data.response, sender: 'bot' };
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.warn("Chatbot connectivity failed.", error);
      const fallbackText = getLocalResponse(userMsgText);
      const botMessage = { id: Date.now() + 1, text: fallbackText, sender: 'bot' };
      setMessages(prev => [...prev, botMessage]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="chatbot-container">
      {/* Floating Button */}
      <button className={`chatbot-toggle ${isOpen ? 'active' : ''}`} onClick={() => setIsOpen(!isOpen)}>
        {isOpen ? '✕' : '💬'}
      </button>

      {/* Chat Window */}
      {isOpen && (
        <div 
          className={`chatbot-window ${isDragging ? 'dragging' : ''}`}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
        >
          <div className="chatbot-header">
            <div className="bot-info">
              <span className="bot-avatar">👨‍⚕️</span>
              <div>
                <h4>FractureAI Medical Advisor</h4>
                <div className="status-container">
                  <span className={`status-indicator ${backendStatus}`}></span>
                  <span className="status-text">
                    {backendStatus === 'online' ? 'Online' : 
                     backendStatus === 'offline' ? 'Offline (Local Mode)' : 
                     'Checking...'}
                  </span>
                </div>
              </div>
            </div>
            <button className="close-btn" onClick={() => setIsOpen(false)}>✕</button>
          </div>

          <div className="chatbot-messages">
            {messages.map(msg => (
              <div key={msg.id} className={`message ${msg.sender}`}>
                <div className="message-content">
                  {msg.image && <img src={msg.image} alt="uploaded" className="chat-image-preview" />}
                  {msg.isPDF && <div className="pdf-attachment-bubble">📄 PDF Report Attached</div>}
                  {msg.text}
                </div>
              </div>
            ))}
            {isTyping && (
              <div className="message bot">
                <div className="message-content typing">
                  <span className="dot"></span>
                  <span className="dot"></span>
                  <span className="dot"></span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="drop-overlay">
            <div className="drop-circle">
                <span>📂 Drop Report Here</span>
            </div>
          </div>

          {filePreview && (
            <div className="file-preview-bar">
                {filePreview === 'PDF_DOCUMENT' ? (
                    <span className="type-icon">📄</span>
                ) : (
                    <img src={filePreview} alt="preview" />
                )}
                <span>Ready to analyze {selectedFile?.name.substring(0, 15)}...</span>
                <button onClick={removeFile}>✕</button>
            </div>
          )}

          <div className="chatbot-input">
            <button className="attach-btn" onClick={() => fileInputRef.current?.click()}>
                📎
            </button>
            <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileChange} 
                style={{display: 'none'}} 
                accept="image/*,.pdf"
            />
            <input
              type="text"
              placeholder="Ask about healing or precautions..."
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
            />
            <button onClick={handleSend} disabled={(!inputValue.trim() && !selectedFile) || isTyping}>
              ➤
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default Chatbot;
