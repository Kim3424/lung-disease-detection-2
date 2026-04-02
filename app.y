import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf  # hoặc torch nếu dùng PyTorch
import os

st.set_page_config(page_title="Lung Disease Detection", layout="centered")
st.title("🫁 Lung Disease Detection")
st.write("Upload ảnh X-quang phổi để dự đoán bệnh")

# ------------------- Load model -------------------
@st.cache_resource
def load_model():
    model_path = "model.h5"          # ← thay bằng tên file model thực tế của bạn
    if not os.path.exists(model_path):
        st.error(f"Không tìm thấy model tại {model_path}")
        st.stop()
    return tf.keras.models.load_model(model_path)   # hoặc torch.load() nếu PyTorch

model = load_model()

# ------------------- Preprocessing -------------------
def preprocess_image(image):
    image = image.resize((224, 224))        # thay kích thước phù hợp với model của bạn
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ------------------- Upload & Predict -------------------
uploaded_file = st.file_uploader("Chọn ảnh X-quang phổi (jpg, png, jpeg)", 
                                 type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ảnh đã upload", use_column_width=True)
    
    if st.button("🔍 Dự đoán"):
        with st.spinner("Đang dự đoán..."):
            processed_img = preprocess_image(image)
            prediction = model.predict(processed_img)
            
            # Giả sử model output là probability cho từng class
            # Bạn cần chỉnh lại theo output thực tế của model (binary hay multi-class)
            class_names = ["Normal", "Pneumonia", "COVID-19", "Lung Cancer", ...]  # ← sửa theo class của bạn
            
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction) * 100
            
            st.success(f"**Kết quả:** {predicted_class}")
            st.info(f"Độ tin cậy: {confidence:.2f}%")
            
            # Hiển thị tất cả xác suất (tùy chọn)
            st.bar_chart({class_names[i]: float(prediction[0][i]) for i in range(len(class_names))})
