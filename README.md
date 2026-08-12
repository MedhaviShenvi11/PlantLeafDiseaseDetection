# 🌿 AgriVision AI - Plant Leaf Disease Detection

AgriVision AI is an AI-powered web application that detects plant leaf diseases from uploaded images using a deep learning model.

The application allows users to upload a plant leaf image and receive an AI-based disease prediction along with the detected crop, model confidence, symptoms, prevention methods, and recommended steps.

---

## 🚀 Features

- 🌿 AI-based plant disease detection
- 📷 Upload plant leaf images
- 🤖 Deep learning-based prediction
- 📊 Model confidence score
- 🔍 Disease symptoms information
- 🛡️ Prevention recommendations
- 🌱 Recommended treatment steps
- ⚡ Fast prediction results
- 📱 Responsive web interface
- 📚 Support for multiple crops
- 🔄 Option to detect another leaf

---

## 🌾 Supported Crops

The application currently supports plant disease detection for:

- 🌾 Rice
- 🍅 Tomato
- 🥔 Potato
- 🌽 Corn

---

## 🦠 Disease Detection

The application can identify multiple diseases depending on the trained model and available dataset.

Examples include:

### Rice
- Brown Spot
- Leaf Smut
- Bacterial Blight

### Tomato
- Early Blight
- Late Blight
- Leaf Mold

### Potato
- Early Blight
- Late Blight
- Healthy Leaf

### Corn
- Common Rust
- Gray Leaf Spot
- Northern Leaf Blight

---

## 🛠️ Technologies Used

### Frontend

- HTML5
- CSS3
- Bootstrap
- JavaScript
- Bootstrap Icons

### Backend

- Python
- Flask

### Machine Learning

- TensorFlow / Keras
- Deep Learning
- Convolutional Neural Network (CNN)
- Image Classification

### Development Tools

- Visual Studio Code
- Git
- GitHub

---

## 📂 Project Structure

```text
PlantLeafDiseaseDetection/
│
├── app.py
├── config.py
├── model.h5
├── labels.txt
├── requirements.txt
├── README.md
│
├── static/
│   ├── css/
│   │   └── style.css
│   │
│   ├── images/
│   │   ├── crops/
│   │   └── hero/
│   │
│   └── uploads/
│
├── templates/
│   ├── layout.html
│   ├── index.html
│   ├── about.html
│   ├── contact.html
│   ├── detect.html
│   ├── result.html
│   │
│   └── errors/
│       ├── 404.html
│       └── 500.html
│
├── utils/
│
└── venv/