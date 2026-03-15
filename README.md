# FractureAI: Bone Fracture Detection using Deep Learning

FractureAI is a medical imaging system that detects bone fractures from X-ray images using **ResNet50** and provides professional medical guidance using **Google Gemini AI**.

---

## 🚀 One-Click Portable Setup (Docker) - RECOMMENDED
This is the **most stable way** to run the project. It handles all dependencies (Python, TensorFlow, Node.js) automatically inside containers.

1.  **Install Docker Desktop**: [Download for Windows](https://www.docker.com/products/docker-desktop)
2.  **Get Gemini API Key**: [Get it at Google AI Studio](https://aistudio.google.com/app/apikey)
3.  **Run with One Command**:
    Double-click or run in terminal:
    ```bash
    .\run-portable.bat
    ```
    *The script will ask for your API key, build the containers, and launch the dashboard at `http://localhost:3000`.*

---

## 🛠️ Developer Local Setup (Manual)
If you prefer running without Docker (Note: Local ResNet models require **Python 3.10-3.11**):

1.  **Install Dependencies**: Run `.\setup.bat` (Sets up virtual environment).
2.  **Configure API Key**: Add `GEMINI_API_KEY=your_key` to `backend/.env`.
3.  **Launch Servers**: Run `.\run.bat` (Starts React and Django simultaneously).

---

## ✨ Features & Architecture
- **Smart API Switching**: The frontend automatically detects if it's running locally (via Docker/Localhost) or in production (via Render) and connects to the appropriate backend.
- **Priority Detection**: The system prioritizes your **trained ResNet50 models** for maximum accuracy.
- **Intelligent Chatbot**: Powered by Gemini 2.0, providing professional orthopedic advice and recovery timelines.
- **Persistent Data**: Results are saved in a local SQLite database and persist across Docker restarts.

## 📁 Project Structure
- `/backend`: Django REST API + AI Inference Engine.
- `/bone-fracture-web`: React.js Modern Dashboard.
- `docker-compose.yml`: Multi-container orchestration.
- `run-portable.bat`: Portable Windows runner.
- `weights/`: Trained H5 model weights for ResNet50 detection.
