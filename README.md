# 🧠 Skin Condition Predictor 🩺 | CNN · ResNet50 · EfficientNetB0

> A deep learning project built to automatically classify six common skin conditions using real-world images. This project leverages transfer learning and CNN architectures to achieve high precision in dermatological diagnostics.

---

## 🔬 Project Overview

This project aims to develop an image classification model that accurately predicts one of **six skin conditions** from input images:

- **Acne**
- **Carcinoma**
- **Eczema**
- **Keratosis**
- **Milia**
- **Rosacea**

📁 The dataset consists of:
- 6 folders (one per class)
- 399 images per class (2,394 total)
- Labeled and sorted into folders

---

## 🧰 Tools & Libraries

| Tool / Library        | Purpose                             |
|-----------------------|-------------------------------------|
| `TensorFlow` / `Keras`| Model building & training           |
| `Google Colab`        | Model development & experimentation |
| `Matplotlib / Seaborn`| Visualization of performance         |
| `Sklearn`             | Evaluation metrics                  |

---

## 📊 Dataset Preprocessing & Augmentation

- Loaded using `image_dataset_from_directory`
- Split: **70% train**, **15% validation**, **15% test**
- Augmentations applied: 
  - Horizontal flip
  - Rotation
  - Zoom
  - Contrast adjustment

---

## 🧠 Models Explored

### 1️⃣ Baseline CNN (Custom Convolutional Neural Network)

🧱 **Architecture**
```plaintext
Conv2D → MaxPooling → Conv2D → MaxPooling → Flatten → Dense → Output
```
- 📈 Performance
- 🧪 Test Accuracy: 83.33%
- 📉 Test Loss: 0.6143
- 🕐 Training Time: ~13s
- 🟡 Comment: Basic CNN showed decent performance but struggled with generalizing on real-life images. Serves well as a foundational benchmark.

### 2️⃣ ResNet50 (Transfer Learning)
⚙️ **Architecture**
```plaintext
Used pre-trained ResNet50 as feature extractor
GlobalAveragePooling2D → Dense(128) → Dropout → Softmax
```
- 🧪 Performance
- ✅ Test Accuracy: 96.61%
- 📉 Test Loss: 0.1118
- 🕐 Training Time: ~116s
- 🟢 Comment: Excellent performance. Handled subtle differences between conditions. A few real-world samples were misclassified.

### 3️⃣ EfficientNetB0 (Fine-Tuned Transfer Learning ✅ Best Performer)
⚙️ **Architecture**
```plaintext
Used EfficientNetB0 as base model
Fully fine-tuned with:
GlobalAvgPool → Dense(128) → Dropout(0.3) → Dense(6)
```
- 🧪 Performance
- 🥇 Test Accuracy: 98.44%
- 📉 Test Loss: 0.0480
- 🕐 Training Time: ~33s
- 🌟 Comment: Best-performing model — highest accuracy, robust generalization, excellent prediction on unseen real-world images.

## 🔍 Evaluation Highlights

| **Model**            | **Test Accuracy** | **Test Loss** | **Real-world Performance**                   |
| -------------------- | ----------------: | ------------: | -------------------------------------------- |
| 🧱 **CNN**           |            83.33% |        0.6143 | ❌ Poor generalization                        |
| 🌀 **ResNet50**      |            96.61% |        0.1118 | ✅ Great accuracy, minor misclassifications   |
| ⚡ **EfficientNetB0** |        **98.44%** |    **0.0480** | 🏆 Best performer – near-perfect predictions |

## 🖼️ Visual Results
📷 Sample Prediction Display
✅ Pred: Eczema       ✅ Pred: Rosacea       ✅ Pred: Milia

## 📉 Training History
- Visualized with accuracy/loss curves
- Used EarlyStopping and ReduceLROnPlateau callbacks
- EfficientNet converged fastest with highest peak performance

## 🧪 External Image Testing
✅ EfficientNetB0 showed high generalization with:
- 💯 High confidence predictions
- 🎯 Accurate classification
- 🔍 Consistency across all 6 skin conditions

## 💾 Model Export Options
- ✅ .keras – Recommended format for modern deployment
- ✅ .h5 – HDF5 legacy format (optional)
- 🔜 Optional: Convert to TensorFlow Lite for mobile devices

## 📚 Future Enhancements
- ✅ Add rare skin conditions
- ✅ Integrate dermoscopic images
- ✅ Implement Grad-CAM or SHAP for interpretability

## 🙌 Credits
| Role            | Person / Tool                         |
| --------------- | ------------------------------------- |
| 👤 Project Lead | *Joshua Oluwole                       |
| 🗂️ Dataset     | Custom folder-organized image dataset |
| ⚙️ Framework    | TensorFlow + Keras                    |
| 📈 Tools        | Sklearn, Matplotlib, Seaborn          |

## 🚀 Run This Project Locally
```plaintext
git clone https://github.com/McNoblesse/Skin_Condition_Predictor.git
cd Skin_Condition_Predictor
python classify_skin_conditions.py
```

## 🧠 Let's Connect
💬 Got questions, ideas or contributions?
- Reach out via email: nobleinepth@gmail.com

**Made with 💙 using TensorFlow & Deep Learning**



