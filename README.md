# 🩺 DermaSaathi: AI-Powered Skin Cancer Detection Web App

**DermaSaathi** is a bias-aware, AI-powered web application built using Streamlit and PyTorch that enables users to perform at-home skin cancer detection. The system uses two deep learning models to:
- Classify skin lesions as **Benign** or **Malignant**
- Predict one of **78 dermatological conditions** from an uploaded lesion image

Built with healthcare equity in mind, DermaSaathi is designed for diverse skin tones and aims to close the diagnostic gap in underserved regions like rural India.

---

## 🚀 Features

- 🧠 **Malignancy Classifier** (ResNet-18 + Attention Layer)
- 🧾 **Lesion Diagnosis System** (ViT-B/16 + Squeeze-Excitation)
- 📷 Upload a lesion image and receive AI-based risk classification
- ⚙️ Lightweight, runs on CPU
- 🌐 Web-based, mobile-friendly via Streamlit

---

## 📁 Project Structure

```
DermaSaathi/
├── app.py                      # Streamlit app
├── resnet_attention_model.pth # Malignancy classifier weights
├── vit_diagnosis_model.pth    # Lesion diagnosis model weights
├── README.md                  # This file
```

---

## 🔧 Setup Instructions

### 1. Install Requirements

Make sure Python 3.8+ is installed, then run:

```bash
pip install streamlit torch torchvision pillow
```

### 2. Run the App

From the project directory:

```bash
streamlit run app.py
```

---

## 🖼️ Input Image Requirements

- Accepted formats: `.jpg`, `.jpeg`, `.png`
- Recommended: clear, close-up photos of skin lesions
- Images are resized to `224x224` before being passed to the models

---

## 🧠 Model Architectures

### 🔬 Malignancy Classifier
- Model: `ResNet-18` with a custom self-attention layer
- Task: Binary classification — `Benign` vs `Malignant`

### 🧾 Lesion Diagnosis System
- Model: Vision Transformer (`ViT-B/16`) + Squeeze-Excitation
- Task: Multiclass classification (78 conditions)

> You can update the class labels inside the app for clearer condition names.

---

## ⚠️ Disclaimer

**This is a prototype intended for demonstration and educational purposes only.**  
It is not a substitute for professional medical advice. Please consult a licensed dermatologist for diagnosis and treatment.

---

## 🛠️ Future Enhancements

- Firebase authentication and scan history
- Telehealth booking integration
- Anonymous user support forums
- Multilingual UI (for Indian regional languages)
- Deployable on Streamlit Cloud

---

## 🤝 Credits

- Developed by **Team MLISM**
- Inspired by Stanford AIMI Center and the DDI dataset
- Built using PyTorch, Streamlit, and open-access medical data

---

## 📄 License

This project is open-source under the MIT License.  
Feel free to fork, adapt, and use with attribution.
