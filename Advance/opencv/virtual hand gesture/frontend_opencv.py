# File: frontend_opencv_frame.py
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.set_page_config(page_title=" Virtual Hand Gesture - OpenCV", layout="wide")
st.title(" Virtual Hand Gesture Tracking (OpenCV + MediaPipe)")

st.markdown("""
This app uses **MediaPipe** and **OpenCV** to track your hand gestures in real time.  
Use this as a frontend for your `opencv_frame.py` script.
""")

# Sidebar Controls
st.sidebar.header(" Controls")
use_webcam = st.sidebar.checkbox("Enable Webcam", value=True)
max_hands = st.sidebar.slider("Max Hands", 1, 2, 1)
detection_conf = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.7)
tracking_conf = st.sidebar.slider("Tracking Confidence", 0.1, 1.0, 0.7)
show_landmarks = st.sidebar.checkbox("Show Landmarks", value=True)
show_connections = st.sidebar.checkbox("Show Connections", value=True)

# Setup Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(max_num_hands=max_hands,
                       min_detection_confidence=detection_conf,
                       min_tracking_confidence=tracking_conf)

# Streamlit placeholder
stframe = st.empty()

if use_webcam:
    cap = cv2.VideoCapture(0)
    st.info(" Webcam is active. Raise your hand to start tracking!")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning(" Unable to access webcam.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if show_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS if show_connections else None,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )

        # Overlay text
        cv2.putText(frame, "Virtual Hand Gesture Active", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Convert and show in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

    cap.release()
    st.success("Webcam stopped.")
else:
    st.warning(" Enable webcam from the sidebar to start tracking.")
