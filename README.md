# Prescription-reader
Perfect. Below is a **clean, professional README** you can **copy-paste directly** for your project that uses **TrOCR (local OCR)** + **Groq API (AI analysis)**.

No emojis, no fluff, college / GitHub / resume ready.

---

# MediScan AI

**Medical Prescription Reader and Intelligent Medicine Recommendation System**

---

## Overview

MediScan AI is a medical prescription analysis system designed to extract handwritten or printed text from prescription images and provide structured, clinically safe insights. The system combines **local OCR using Microsoft TrOCR** with **AI-based reasoning using the Groq API**, ensuring accuracy, privacy, and cost efficiency.

The application is intended for **educational and research purposes only** and does not replace professional medical advice.

---

## Key Features

* Handwritten prescription text extraction using **local TrOCR (CPU-based)**
* No dependency on paid OCR services
* AI-powered prescription understanding using **Groq LLM**
* Detection of condition category (infection, fever, pain, nutritional support, etc.)
* Structured medicine extraction (name, strength, frequency, duration)
* Safety-first design:

  * No dosage guessing
  * No frequency conversion (TID, BD, OD preserved)
  * Unclear text explicitly marked
* Disease-based medicine reference search
* Clear medical disclaimers and warnings
* Works fully on CPU (no GPU required)

---

## System Architecture

```
Prescription Image
        ↓
Local OCR (Microsoft TrOCR)
        ↓
Raw Extracted Text
        ↓
Groq LLM Analysis
        ↓
Structured Medical Output
        ↓
Safe Medicine Reference Information
```

---

## Technologies Used

* **Python**
* **Flask** – Backend server
* **Microsoft TrOCR** – Local handwritten OCR
* **Groq API** – Medical text analysis and reasoning
* **Transformers (HuggingFace)**
* **OpenCV & Pillow** – Image preprocessing
* **HTML/CSS/JavaScript** – Frontend UI

---

## Why TrOCR + Groq?

### TrOCR (Local OCR)

* Completely free
* Works offline
* Designed for handwritten text
* No billing or API limits
* Ideal for doctor handwriting

### Groq API

* High-speed large language models
* Used only for reasoning, not OCR
* No medical hallucination when prompted correctly
* Free tier sufficient for academic use

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/mediscan-ai.git
cd mediscan-ai
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install transformers flask flask-cors python-dotenv pillow opencv-python requests
```

---

## Environment Variables

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_api_key_here
```

You can obtain the key from:
[https://console.groq.com](https://console.groq.com)

---

## Running the Application

```bash
python app.py
```

Open in browser:

```
http://localhost:5000
```

---

## Supported Inputs

* JPG / PNG images
* Printed or handwritten prescriptions
* Clear photos with good lighting recommended

---

## Safety and Medical Disclaimer

This project is strictly for **educational and research purposes**.

* Does NOT provide medical advice
* Does NOT replace a licensed doctor
* Does NOT recommend medicine substitutions
* Antibiotics and prescription drugs must only be taken as prescribed by a doctor

Always consult a qualified healthcare professional.

---

## Known Limitations

* Handwritten OCR accuracy depends on image quality
* Very messy handwriting may produce partial results
* AI analysis depends on OCR quality
* Not intended for real-world clinical deployment

---

## Future Improvements

* Spell correction for medicine names
* Confidence scoring per extracted field
* Multilingual prescription support
* Integration with drug interaction databases
* Improved UI feedback for unclear text
* Optional offline LLM integration

---

## License

This project is released under the **MIT License**.

