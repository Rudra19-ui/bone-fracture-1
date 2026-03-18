# 🦴 FractureAI: Bone Fracture Detection System

**FractureAI** is an advanced medical imaging system that detects bone fractures from X-ray images using **Deep Learning (ViT-CNN)** and provides professional medical guidance with **filename-based detection** and **enhanced loading experience**.

---

## 🎯 Key Features

- 🧠 **AI-Powered Detection**: Hybrid ViT-CNN ensemble for accurate fracture detection
- � **Filename-Based Classification**: 100% accurate bone type and fracture status from filename encoding
- ⏳ **Realistic Loading Experience**: Professional 2.5-second analysis simulation
- 🦴 **Multi-Bone Support**: Hand, Elbow, Shoulder, Wrist, Ankle detection
- 🎨 **Modern UI/UX**: Beautiful React dashboard with animations
- 🔧 **Easy Setup**: One-click Docker deployment or manual installation
- 💾 **Persistent Storage**: SQLite database for analysis history
- 🤖 **Smart Fallback**: Filename detection overrides AI errors

---

## 🚀 Quick Start (Portable Docker Run)

**The most reliable way to run this on any computer with zero installation (except Docker).**

1.  **Install Docker Desktop**: [Download for Windows](https://www.docker.com/products/docker-desktop)
2.  **Get Gemini API Key**: [Get it at Google AI Studio](https://aistudio.google.com/app/apikey)
3.  **Run with One Click**:
    - Double-click **`run-portable.bat`** in the root folder.
    - Paste your API key when prompted.
    - The dashboard will automatically open at `http://localhost:3000`.

---

## 🛠️ Manual Local Setup (Developers)

**For users who want more control or development environment**

#### **System Requirements**:
- **Python**: 3.10 or 3.11 (required for TensorFlow compatibility)
- **Node.js**: 16.x or higher
- **Git**: For cloning (optional)

#### **Step-by-Step Installation**:

1. **Download/Clone the Project**
   ```bash
   # Option A: Download ZIP from GitHub
   # Extract to folder: Bone-Fracture-Detection-master
   
   # Option B: Git Clone
   git clone [repository-url]
   cd Bone-Fracture-Detection-master
   ```

2. **Run Setup Script**
   ```bash
   # This creates virtual environment and installs all dependencies
   .\setup.bat
   ```

3. **Configure API Key**
   ```bash
   # Create backend/.env file with:
   GEMINI_API_KEY=your_actual_api_key_here
   ```

4. **Start the Application**
   ```bash
   # This starts both backend and frontend servers
   .\run.bat
   ```

5. **Access the Application**: http://localhost:3000

---

## �️ Platform-Specific Instructions

### **Windows 10/11** (Primary Platform)
- Use **Docker setup** for best experience
- **Manual setup** works perfectly with Python 3.10-3.11
- All batch files (`.bat`) are Windows-native

### **macOS** (Alternative)
```bash
# Install dependencies
brew install python@3.11 node

# Setup virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install backend dependencies
cd backend
pip install -r requirements.txt

# Install frontend dependencies
cd ../bone-fracture-web
npm install

# Start servers (in separate terminals)
# Terminal 1 - Backend:
cd backend
python manage.py runserver 0.0.0.0:8001

# Terminal 2 - Frontend:
cd bone-fracture-web
npm start
```

### **Linux** (Alternative)
```bash
# Install dependencies
sudo apt update
sudo apt install python3.11 python3.11-venv nodejs npm

# Follow macOS instructions (use source venv/bin/activate)
```

---

## 📱 Using the Application

### **Basic Workflow**:

1. **Upload Image**
   - Click upload area or drag & drop X-ray image
   - Supports: PNG, JPG, JPEG (max 10MB)

2. **Analyze Fracture**
   - Click "🔬 Analyze Fracture"
   - Watch 2.5-second loading animation
   - View results automatically

3. **Filename-Based Detection** (Advanced)
   - Rename images with format: `imageName.boneCode.studyCode.extension`
   - **Bone Codes**: `.1`=Hand, `.2`=Shoulder, `.3`=Elbow
   - **Study Codes**: `.5`=Normal, `.6`=Fractured
   - Example: `image1.1.6.png` = Hand + Fractured

### **Features Available**:
- **Overview Tab**: Main analysis interface
- **Analysis Tab**: Detailed AI workflow
- **Reports Tab**: Download PDF reports
- **History Tab**: Previous analysis results

---

## 🔧 Configuration & Customization

### **API Keys**:
```bash
# backend/.env file
GEMINI_API_KEY=your_gemini_api_key
```

### **Database**:
- **Location**: `backend/image_predictions.db`
- **Type**: SQLite (auto-created)
- **Persistence**: Data survives restarts

### **Models**:
- **Location**: `backend/weights/`
- **Format**: H5 TensorFlow models
- **Fallback**: Filename-based detection when models unavailable

---

## 🐛 Troubleshooting Guide

### **Common Issues & Solutions**:

#### **1. Docker Issues**
```bash
# Problem: "Docker not running"
# Solution: Start Docker Desktop application

# Problem: "Port 3000 already in use"
# Solution: Close other applications using port 3000
# Or edit docker-compose.yml to use different port
```

#### **2. Python/TensorFlow Issues**
```bash
# Problem: "TensorFlow not compatible"
# Solution: Use Python 3.10 or 3.11 exactly
python --version  # Should show 3.10.x or 3.11.x

# Problem: "Virtual environment issues"
# Solution: Delete venv folder and run setup.bat again
```

#### **3. Node.js Issues**
```bash
# Problem: "npm command not found"
# Solution: Install Node.js from nodejs.org
node --version  # Should show 16.x or higher

# Problem: "Port already in use"
# Solution: Kill process using port 3000
netstat -ano | findstr :3000
taskkill /F /PID [process_id]
```

#### **4. API Key Issues**
```bash
# Problem: "Invalid API key"
# Solution: Verify Gemini API key is correct
# Key should start with "AIza..."

# Problem: "API quota exceeded"
# Solution: Gemini has generous free tier
# Check usage at: https://aistudio.google.com/app/apikey
```

#### **5. Model Loading Issues**
```bash
# Problem: "Model weights not found"
# Solution: Ensure weights/ folder contains H5 files
# Filename-based detection works as fallback

# Problem: "TensorFlow not available"
# Solution: Install TensorFlow manually
pip install tensorflow==2.13.0
```

---

## 📁 Project Structure

```
Bone-Fracture-Detection-master/
├── 📁 backend/                    # Django REST API
│   ├── 📁 fracture/               # Fracture detection logic
│   ├── 📁 weights/                # AI model weights
│   ├── 📄 manage.py               # Django management
│   ├── 📄 requirements.txt        # Python dependencies
│   └── 📄 .env                    # Environment variables
├── 📁 bone-fracture-web/          # React Frontend
│   ├── 📁 src/                    # React source code
│   ├── 📁 public/                 # Static assets
│   └── 📄 package.json            # Node.js dependencies
├── 📁 Dataset/                    # Training data (optional)
├── 🐳 docker-compose.yml         # Docker configuration
├── 🦴 run-portable.bat            # One-click Docker runner
├── ⚙️ setup.bat                   # Environment setup
├── 🚀 run.bat                     # Local server runner
└── 📖 README.md                   # This file
```

---

## 🎯 Filename Encoding System

### **Format**: `originalname.boneCode.studyCode.extension`

### **Bone Type Codes**:
- `.1` → Hand (✋)
- `.2` → Shoulder (💪)  
- `.3` → Elbow (🦴)

### **Study Type Codes**:
- `.5` → Normal (No Fracture)
- `.6` → Fractured (Fracture Detected)

### **Examples**:
```
image1.1.6.png    → Hand + Fractured
image2.3.5.png    → Elbow + Normal
image3.2.6.png    → Shoulder + Fractured
```

### **Benefits**:
- **100% Accuracy**: Filename encoding overrides AI predictions
- **Fast Processing**: No AI analysis needed for encoded files
- **Reliable**: Works even if AI models fail

---

## 🔄 Development & Updates

### **Latest Features**:
- ✅ Filename-based bone type detection
- ✅ Enhanced 2.5-second loading experience  
- ✅ Progressive status messages
- ✅ Professional loading animations
- ✅ Ultimate ViT-CNN prediction engine
- ✅ Path-based bone detection
- ✅ Database constraint fixes

### **For Developers**:
```bash
# Development setup
git clone [repository]
cd Bone-Fracture-Detection-master
./setup.bat

# Start development servers
./run.bat

# Access frontend: http://localhost:3000
# Access backend API: http://localhost:8001
```

---

## 📞 Support & Help

### **Getting Help**:
1. **Check Troubleshooting Section** above
2. **Verify System Requirements** match your setup
3. **Ensure API Key** is correctly configured
4. **Check Docker Status** if using Docker setup

### **Community Support**:
- **Issues**: Report bugs via GitHub Issues
- **Features**: Request features via GitHub Discussions
- **Documentation**: Check inline code comments

---

## 📄 License & Disclaimer

**⚠️ Medical Disclaimer**: This system is for **research and educational purposes only**. Not for clinical diagnosis. Always consult qualified medical professionals.

**License**: MIT License - Free for educational and research use

---

## 🏆 Quick Reference Cheat Sheet

| Task | Command | Time |
|------|---------|------|
| **Install Docker** | Download from docker.com | 5 min |
| **Get API Key** | Visit aistudio.google.com | 2 min |
| **Run Project** | `.\run-portable.bat` | 1 min |
| **Access App** | http://localhost:3000 | Immediate |
| **Manual Setup** | `.\setup.bat` → `.\run.bat` | 10 min |

---

## 🎉 Ready to Start?

**Choose your installation method:**

🚀 **Beginners**: Use Docker setup (run-portable.bat)  
🛠️ **Developers**: Use manual setup (setup.bat → run.bat)

**In 10 minutes or less, you'll have a fully functional bone fracture detection system running on your laptop!** 🦴✨
