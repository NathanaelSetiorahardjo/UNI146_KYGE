import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import requests  # ✅ NEW: For Ubidots
import time       # ✅ NEW: To manage send intervals

# === Streamlit Page Setup (must be first) ===
st.set_page_config(page_title="🖐 BISINDO Sign Recognition", layout="wide")

# === Title Section ===
st.markdown("""
    <h1 style='text-align: center;'>🖐 BISINDO Sign Recognition</h1>
    <p style='text-align: center; font-size: 18px;'>
        Real-time A-Z sign recognition powered by ESP32-CAM + TensorFlow. Built with ❤ using Streamlit.
    </p>
""", unsafe_allow_html=True)
st.divider()

# === Load Keras Model (cached) ===
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("bisindo_model.h5")

model = load_model()
class_labels = [chr(i) for i in range(65, 91)]  # A-Z
img_size = (96, 96)

# === MediaPipe Hands Setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# === Sidebar: Settings ===
with st.sidebar:
    st.header("⚙ Settings")
    stream_url = st.text_input("ESP32-CAM Stream URL", "http://192.168.101.7:81/stream")
    run = st.checkbox("▶ Start Recognition")

# === Main Layout: 3 columns, center aligned content ===
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    video_placeholder = st.empty()
    prediction_placeholder = st.empty()
    chart_placeholder = st.empty()

# === Helper Function: Plot confidence bar chart ===
def plot_confidences(confidences, labels):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels, confidences, color='skyblue')
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel("Confidence")
    ax.set_title("Model Prediction Confidence")
    return fig

# === ✅ Ubidots Config and Send Function ===
UBIDOTS_TOKEN = "BBUS-nVBcjWjcD0R6gTWTlf2UxAeGMRHO5I"
DEVICE_LABEL = "prototype"
UBIDOTS_URL = f"https://industrial.api.ubidots.com/api/v1.6/devices/{DEVICE_LABEL}/"
HEADERS = {
    "X-Auth-Token": UBIDOTS_TOKEN,
    "Content-Type": "application/json"
}

def send_to_ubidots(predicted_letter, confidence):
    if predicted_letter is None or predicted_letter == "" or confidence is None:
        print("🚫 Skipping Ubidots send due to empty prediction.")
        return

    payload = {
        "predicted_label": ord(predicted_letter) - ord('A'),           # Send Letter Code
        "confidence": int(confidence * 100)
    }

    try:
        response = requests.post(UBIDOTS_URL, headers=HEADERS, json=payload)

        # UBIDOTS DEBUGS
        if response.status_code not in [200, 201]:
            print(f"⚠️ Ubidots send failed: {response.status_code} - {response.text}")
        else:
            print(f"✅ Ubidots sent     : {payload}")
            print(f"✅ Predicted letter : {predicted_letter}")
    except Exception as e:
        print(f"❌ Ubidots exception: {e}") 



last_sent = 0  # For throttling

# === Start Recognition if toggled ===
if run:
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        st.error("❌ Failed to open ESP32-CAM stream.")
    else:
        while run:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            prediction_text = ""
            confidence_array = None

            if results.multi_hand_landmarks:
                x_coords_all, y_coords_all = [], []
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        h, w, _ = frame.shape
                        x_coords_all.append(int(lm.x * w))
                        y_coords_all.append(int(lm.y * h))
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                x_min = max(min(x_coords_all) - 20, 0)
                x_max = min(max(x_coords_all) + 20, frame.shape[1])
                y_min = max(min(y_coords_all) - 20, 0)
                y_max = min(max(y_coords_all) + 20, frame.shape[0])

                combined_hand_img = frame[y_min:y_max, x_min:x_max]
                if combined_hand_img.size != 0:
                    resized = cv2.resize(combined_hand_img, img_size)
                    normalized = resized.astype("float32") / 255.0
                    input_tensor = np.expand_dims(normalized, axis=0)

                    prediction = model.predict(input_tensor, verbose=0)
                    class_index = np.argmax(prediction)
                    confidence = prediction[0][class_index]
                    predicted_label = class_labels[class_index]
                    confidence_array = prediction[0]

                    # Show prediction label
                    prediction_html = f"<h3 style='text-align: center;'>Predicted: {predicted_label} ({confidence:.2f})</h3>"
                    prediction_placeholder.markdown(prediction_html, unsafe_allow_html=True)

                    # ✅ Send to Ubidots every 5 seconds
                    if time.time() - last_sent > 5:
                        send_to_ubidots(predicted_label, confidence)
                        last_sent = time.time()

            # Show video stream with updated keyword
            video_placeholder.image(frame, channels="BGR", use_container_width=True)

            # Show confidence bar chart
            if confidence_array is not None:
                sorted_indices = np.argsort(confidence_array)[::-1][:5]  # top 5
                top_labels = [class_labels[i] for i in sorted_indices]
                top_confidences = [confidence_array[i] for i in sorted_indices]
                fig = plot_confidences(top_confidences, top_labels)
                chart_placeholder.pyplot(fig)

        cap.release()
else:
    prediction_placeholder.info("Check ▶ Start Recognition on the sidebar to begin.")
