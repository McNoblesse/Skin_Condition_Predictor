# 🧠 Skin Condition Predictor 🩺 | · CNN · | ·  ResNet50 · | · EfficientNetB0 · + Streamlit App 🎨

> A deep learning project built to automatically classify six common skin conditions using real-world images. This project leverages transfer learning, CNN architectures, and an interactive Streamlit web app to achieve high precision in dermatological diagnostics and user accessibility.

---

## 🎓 Project Overview

This project aims to develop an image classification model that accurately predicts one of **six skin conditions** from input images:

* **Acne**
* **Carcinoma**
* **Eczema**
* **Keratosis**
* **Milia**
* **Rosacea**

📁 The dataset consists of:

* 6 folders (one per class)
* 399 images per class (2,394 total)
* Labeled and sorted into folders

---

## 🛠️ Tools & Libraries

| Tool / Library         | Purpose                             |
| ---------------------- | ----------------------------------- |
| `TensorFlow` / `Keras` | Model building & training           |
| `Streamlit`            | Interactive web application         |
| `Google Colab`         | Model development & experimentation |
| `Matplotlib / Seaborn` | Visualization of performance        |
| `Sklearn`              | Evaluation metrics                  |

---

## 📊 Dataset Preprocessing & Augmentation

* Loaded using `image_dataset_from_directory`
* Split: **70% train**, **15% validation**, **15% test**
* Augmentations applied:

  * Horizontal flip
  * Rotation
  * Zoom
  * Contrast adjustment

---

## 🧠 Models Explored

### 1️⃣ Baseline CNN (Custom Convolutional Neural Network)

🧱 **Architecture**

```
Conv2D → MaxPooling → Conv2D → MaxPooling → Flatten → Dense → Output
```

* 📈 Test Accuracy: **83.33%**
* 📉 Test Loss: **0.6143**
* ⏱ Training Time: \~13s
* 🟡 Comment: Basic CNN showed decent performance but struggled with generalizing on real-life images. Serves well as a foundational benchmark.

### 2️⃣ ResNet50 (Transfer Learning)

⚙️ **Architecture**

```
Pre-trained ResNet50 → GlobalAvgPool → Dense(128) → Dropout → Softmax
```

* 📈 Test Accuracy: **96.61%**
* 📉 Test Loss: **0.1118**
* ⏱ Training Time: \~116s
* 🟢 Comment: Excellent performance. Handled subtle differences between conditions. A few real-world samples were misclassified.

### 3️⃣ EfficientNetB0 (Fine-Tuned Transfer Learning) ✅ **Best Performer**

💡 **Architecture**

```
Pre-trained EfficientNetB0 → GlobalAvgPool → Dense(128) → Dropout(0.3) → Dense(6)
```

* 📈 Test Accuracy: **98.44%**
* 📉 Test Loss: **0.0480**
* ⏱ Training Time: \~33s
* 🌟 Comment: Best-performing model — highest accuracy, robust generalization, excellent prediction on unseen real-world images.

---

## 🌍 Interactive Streamlit App

The project includes a **user-friendly Streamlit app** that enables:

* 📄 Image upload from local machine
* 🧠 Predictions using EfficientNetB0 model
* 📊 Top-3 class confidence display
* 🌍 Beautiful layout with emojis and clean interface

### 🚀 Run the App

```bash
git clone https://github.com/McNoblesse/Skin_Condition_Predictor.git
cd Skin_Condition_Predictor
streamlit run app.py
```

> Ensure the model file `efficientnet_model_skin_condition.keras` is placed in the root directory.

---

## 🔍 Evaluation Highlights

| **Model**            | **Test Accuracy** | **Test Loss** | **Real-world Performance**                   |
| -------------------- | ----------------: | ------------: | -------------------------------------------- |
| 🧱 **CNN**           |            83.33% |        0.6143 | ❌ Poor generalization                        |
| 🌀 **ResNet50**      |            96.61% |        0.1118 | ✅ Great accuracy, minor misclassifications   |
| ⚡ **EfficientNetB0** |        **98.44%** |    **0.0480** | 🏆 Best performer – near-perfect predictions |

---

## 🖼️ Visual Results

📷 Sample Prediction Display:

* ✅ Pred: Eczema
* ✅ Pred: Rosacea
* ✅ Pred: Milia

📊 Training History:

* Accuracy/Loss curves
* EarlyStopping and ReduceLROnPlateau used
* EfficientNet converged fastest

🔮 External Image Testing:

* ✅ High confidence
* ✅ Consistent results
* ✅ Tested on real-world unseen data

---

## 💾 Model Export Options

* ✅ `.keras` (Recommended modern format)
* ✅ `.h5` (Legacy HDF5 backup)
* ⏳ Optional: TensorFlow Lite conversion (for mobile)

---

## 📖 Future Enhancements

* ✅ Add rare skin conditions
* ✅ Integrate dermoscopic images
* ✅ Streamlit deployment via HuggingFace/Streamlit Cloud
* 🤍 Add Grad-CAM or SHAP for interpretability

---

## 🙌 Credits

| Role            | Person / Tool                         |
| --------------- | ------------------------------------- |
| 👤 Project Lead | *Joshua Oluwole*                      |
| 🗂️ Dataset     | Custom folder-organized image dataset |
| ⚙️ Framework    | TensorFlow + Keras                    |
| 📊 Tools        | Sklearn, Matplotlib, Seaborn          |

---

## 🚀 Local Usage (Script Version)

```bash
git clone https://github.com/McNoblesse/Skin_Condition_Predictor.git
cd Skin_Condition_Predictor
python classify_skin_conditions.py
```

---

## 🧠 Let's Connect

💬 Got questions, ideas, or feedback?

* Email: [nobleinepth@gmail.com](mailto:nobleinepth@gmail.com)
* GitHub Issues welcome!

**Made with 💙 using TensorFlow, Keras, and Streamlit**
