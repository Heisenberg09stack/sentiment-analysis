# 🎭 Emotion & Sentiment Analysis API

An end-to-end Deep Learning and FastAPI project for classifying emotions in text using Recurrent Neural Networks (RNNs, LSTMs, GRUs, and Bidirectional GRUs).

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)

---

## 📌 Project Overview

This repository provides a complete pipeline for emotion recognition from text:
1. **Model Exploration & Benchmarking**: A comprehensive Jupyter Notebook (`final_clean.ipynb`) training and comparing multiple sequence models:
   - **Simple RNN**
   - **Standard LSTM**
   - **Standard GRU**
   - **Stacked Bidirectional GRU (Final Selected Model)**
2. **RESTful API Service**: Built with **FastAPI** (`main.py`) to serve real-time predictions with confidence scores and full probability distributions across 6 emotion classes:
   - 😢 Sadness
   - 😄 Joy
   - ❤️ Love
   - 😠 Anger
   - 😨 Fear
   - 😲 Surprise

---

## 📁 Repository Structure

```
.
├── final_clean.ipynb   # Complete ML Pipeline: EDA, Preprocessing, Benchmarking & Saving Models
├── main.py             # FastAPI backend server with model inference & prediction endpoints
├── requirements.txt    # Python dependencies
├── runtime.txt         # Python runtime version for deployment (e.g. Render, Heroku)
├── .gitignore          # Git ignore configuration
└── README.md           # Project documentation
```

---

## 🛠️ Installation & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Heisenberg09stack/sentiment-analysis.git
cd sentiment-analysis
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Step 1: Model Training & Artifact Generation
Run all cells in `final_clean.ipynb` to train the models and export the trained model & tokenizer into an `Artifacts/` folder:
- `Artifacts/BiGRU_Model.keras`
- `Artifacts/tokenizer.pkl`

### Step 2: Start the FastAPI Server
```bash
uvicorn main:app --reload --port 8000
```

Access the interactive API documentation at:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 🔌 API Endpoints

### 1. Health Check
- **`GET /health`**
- Verifies server status and model load state.

**Response:**
```json
{
  "status": "Server is running",
  "model_loaded": true
}
```

### 2. Emotion Prediction
- **`POST /predict`**
- Input: JSON containing the text payload.

**Request Body:**
```json
{
  "text": "I feel so happy and excited about this breakthrough!"
}
```

**Response:**
```json
{
  "text": "I feel so happy and excited about this breakthrough!",
  "predicted_emotion": "joy",
  "confidence": 0.9842,
  "all_probabilites": {
    "sadness": 0.0021,
    "joy": 0.9842,
    "love": 0.0081,
    "anger": 0.0019,
    "fear": 0.0011,
    "surprise": 0.0026
  }
}
```

---

## ⚙️ Model Architecture Details

The primary model leverages a **Stacked Bidirectional GRU (Gated Recurrent Unit)** network:
- **Text Cleaning**: Lowercasing, punctuation stripping, contractions normalization.
- **Tokenization & Padding**: Max sequence length = 50.
- **Model Layers**:
  - Embedding Layer
  - Bidirectional GRU Layer
  - Dropout & Dense Layers
  - Softmax Output (6 emotion classes)

---

## 👤 Author

**Heisenberg09stack**  
GitHub Profile: [github.com/Heisenberg09stack](https://github.com/Heisenberg09stack)
