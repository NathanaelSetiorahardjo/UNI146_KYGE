import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import paho.mqtt.client as mqtt
import json
import time

UBIDOTS_TOKEN = "BBUS-nVBcjWjcD0R6gTWTlf2UxAeGMRHO5I"
DEVICE_LABEL = "prototype"
BROKER = "industrial.api.ubidots.com"
PORT = 1883
TOPIC_LETTER = f"/v1.6/devices/{DEVICE_LABEL}/prediction"
TOPIC_CONFIDENCE = f"/v1.6/devices/{DEVICE_LABEL}/confidence"

# MQTT Setup
client = mqtt.Client()
client.username_pw_set(UBIDOTS_TOKEN)
client.connect(BROKER, PORT)
client.loop_start()

model = tf.keras.models.load_model("bisindo_model.h5")
class_labels = [chr(i) for i in range(65, 91)]  # A-Z
img_size = (96, 96)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def connect_to_stream():
    cap = cv2.VideoCapture("http://192.168.101.16:81/stream")
    while not cap.isOpened():
        print("❌ Failed to open ESP32-CAM stream. Retrying...")
        time.sleep(3)
        cap = cv2.VideoCapture("http://192.168.101.16:81/stream")
    print("✅ Stream opened.")
    return cap

cap = connect_to_stream()

prev_label = None
prev_confidence = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame read failed. Retrying...")
        cap.release()
        cap = connect_to_stream()
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

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

            letter_value = ord(predicted_label) - ord('A')

            if abs(confidence - prev_confidence) > 0.05:
                prev_label = predicted_label
                prev_confidence = confidence

                rounded_confidence = round(float(confidence), 2)  # ✅ Convert numpy.float32 → float + round

                payload_letter = json.dumps({"value": letter_value})
                payload_confidence = json.dumps({"value": rounded_confidence})

                print(f"📤 Sending to MQTT → letter: {payload_letter}, confidence: {payload_confidence}")

                client.publish(TOPIC_LETTER, payload_letter)
                client.publish(TOPIC_CONFIDENCE, payload_confidence)

            cv2.putText(frame, f"{predicted_label} ({rounded_confidence:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("ESP32-CAM Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("⏹️ Quitting...")
        break

cap.release()
cv2.destroyAllWindows()
client.disconnect()
print("✅ Clean exit.")
