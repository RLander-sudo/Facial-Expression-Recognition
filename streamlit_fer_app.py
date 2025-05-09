#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ---------------------------------------------
# Streamlit Real-Time FER with Webcam
# ---------------------------------------------
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import numpy as np
import tensorflow as tf
import pickle

st.set_page_config(page_title="Real-Time Facial Expression Recognition", layout="centered")

# Load model and class names
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/expression_model_20250509-093652.h5")
    with open("data/class_names.pkl", "rb") as f:
        class_names = pickle.load(f)
    return model, class_names

model, class_names = load_model()
IMG_SIZE = 48

# Define Video Processor
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE)) / 255.0
            face_input = np.expand_dims(face, axis=(0, -1))

            prediction = model.predict(face_input)
            emotion_idx = np.argmax(prediction)
            label = class_names[emotion_idx]
            confidence = np.max(prediction)

            label_text = f"{label} ({confidence*100:.1f}%)"
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, label_text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

# Streamlit UI
st.title("🎥 Real-Time Facial Expression Recognition")

webrtc_streamer(
    key="emotion",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)


# In[ ]:




