# Health-e-AI: Next-Gen IoT-Integrated Healthcare Diagnostics 🏥🚀

An advanced AI-powered diagnostic system that integrates **Machine Learning** and **IoT (Hardware)** to provide rapid, accurate, and accessible healthcare screening.

## 🌟 Key Features

- **Multi-Disease Detection:** Integrated models for Heart, Liver, Kidney, Diabetes, and Cancer diagnostics based on patient health parameters.
- **Medical Imaging Analysis:** Specialized CNN-based models for detecting Pneumonia and Malaria from X-rays and blood smears.
- **IoT Health Monitoring:** Integrated Arduino-based health monitoring system for real-time tracking of vital signs.
- **Unified Diagnostic Engine:** A modular AI core that can be easily extended with new medical models.

## 🛠️ Technical Architecture

### 1. AI Diagnostic Core
The system utilizes a hybrid approach to diagnostics:
- **Tabular Data (ML):** Logistic Regression and Random Forest models for physiological parameters.
- **Medical Imaging (DL):** Convolutional Neural Networks (CNN) for high-accuracy image classification.
- **Tech Stack:** TensorFlow, Keras, Scikit-learn, NumPy, Pandas.

### 2. Web Interface
- **Backend:** Flask (Python)
- **Frontend:** HTML5, CSS3, JavaScript
- **Integration:** Modular API-style routing for seamless disease-specific diagnostics.

### 3. IoT Integration
- **Hardware:** Arduino / ESP32 sensors
- **Functionality:** Real-time data streaming of vital signs (BPM, SpO2, Temp) into the diagnostic pipeline.

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/Ravi-Tiwari-2710/Health-e-AI.git
pip install -r requirements.txt
python web/app.py
```

## 📜 Patent & Recognition
This project was developed as part of a broader vision for assistive healthcare technology, integrating predictive AI for early-stage disease detection.

---
*Developed by [Ravi Tiwari](https://github.com/Ravi-Tiwari-2710)*
