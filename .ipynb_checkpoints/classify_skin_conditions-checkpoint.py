# 🧠 Streamlit App: Skin Condition Predictor (EfficientNetB0)

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model

# ------------------------------------
# 🎯 Config
# ------------------------------------
st.set_page_config(page_title="Skin Condition Predictor", page_icon="🧠", layout="centered")
st.title("🧠 Skin Condition Classifier")
st.markdown("""
Upload an image and classify one of six common skin conditions using a fine-tuned EfficientNetB0 model. 📷💡
""")

# ------------------------------------
# 🏷️ Class Names
# ------------------------------------
CLASS_NAMES = ['Acne', 'Carcinoma', 'Eczema', 'Keratosis', 'Milia', 'Rosacea']

# ------------------------------------
# 📥 Load Model
# ------------------------------------
@st.cache_resource
def load_skin_model():
    model_path = "models/efficientnet_model_skin_condition.keras"  # Replace with your model path
    model = load_model(model_path, custom_objects={'preprocess_input': preprocess_input})
    return model

model = load_skin_model()

# ------------------------------------
# 📷 Upload Image
# ------------------------------------
uploaded_file = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='📷 Uploaded Image', use_column_width=True)

    # Preprocess image
    img_resized = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)

    # Predict
    predictions = model.predict(img_preprocessed)
    confidence_scores = tf.nn.softmax(predictions[0]).numpy()

    pred_index = np.argmax(confidence_scores)
    predicted_class = CLASS_NAMES[pred_index]

    # Results
    st.markdown("---")
    st.subheader("📊 Prediction Result")
    st.success(f"✅ **Predicted:** {predicted_class}")

    st.markdown("### 🔢 Confidence Scores")
    for i, (cls, conf) in enumerate(zip(CLASS_NAMES, confidence_scores)):
        st.write(f"{cls}: {conf * 100:.2f}%")

    # Show as bar chart
    st.bar_chart({CLASS_NAMES[i]: confidence_scores[i] for i in range(len(CLASS_NAMES))})

# Footer
st.markdown("---")
st.markdown("""
### 🧠 About This App
Made with ❤️ using TensorFlow, Keras & Streamlit  
Project by: **Your Name**

- Acne
- Carcinoma
- Eczema
- Keratosis
- Milia
- Rosacea

💡 Built with [Streamlit](https://streamlit.io), TensorFlow, and Keras.

📌 Models trained on custom dataset.

👨‍💻 Developed by Joshua Oluwole
""")
