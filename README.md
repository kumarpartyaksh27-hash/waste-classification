# ♻️ Waste Classification using Deep Learning (CNN)

A Deep Learning project that uses a **Convolutional Neural Network (CNN)** to classify waste images into two categories: **Organic (Wet Waste)** and **Recyclable (Dry Waste)**.

The project is built using **Python, TensorFlow, and Keras** and demonstrates the basic workflow of an image classification system.

---

## 📌 Project Overview

Proper waste segregation plays an important role in recycling and sustainable waste management.

This project explores how Deep Learning and Computer Vision can be used to automatically classify waste based on images.

### 🎯 Target Classes

- 🟢 **Organic (Wet Waste)**
- ♻️ **Recyclable (Dry Waste)**

### 📐 Input Image Size

`128 × 128 × 3`

---

## 🧠 Model Architecture

The project uses a **Convolutional Neural Network (CNN)** consisting of:

- Convolutional layers for feature extraction
- Max Pooling layers for dimensionality reduction
- Flatten layer
- Dropout Regularization to help reduce overfitting
- Dense layers for classification
- Softmax output layer for two-class prediction

---

## 🛠️ Tech Stack

- **Python 3.12**
- **TensorFlow**
- **Keras**
- **NumPy**
- **OpenCV**
- **Matplotlib**
- **Scikit-learn**

---

## 📂 Project Structure

```text
waste-classification/
│
├── train.py
├── predict.py
├── requirements.txt
├── README.md
│
└── dataset/
    ├── Organic/
    └── Recyclable/
