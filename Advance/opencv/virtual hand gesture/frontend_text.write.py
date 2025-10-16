# File: frontend_text_write.py
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from collections import deque

st.set_page_config(page_title=" Virtual Hand Gesture Text Writer", layout="wide")
st.title(" Virtual Hand Gesture - Write or Draw with Hand Tracking")

st.markdown("""
Control your screen using hand gestures!  
Move your **index finger** to draw on the screen in real time.  
Use your **thumb and index finger together** to stop writing.
""")

# Sidebar controls
st.sidebar.header("Controls")
use_webcam = st.sidebar.checkbox("Enable Webcam", value=True)
brush_color = st.sidebar.color_picker("Select Brush Color", "#00FF00")
brush_size = st.sidebar.slider("Brush Size", 3, 20, 6)
eraser_mode = st.sidebar.checkbox("Eraser Mode", value=False)

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

stframe = st.empty()
if use_webcam:
    cap = cv2.VideoCapture(0)
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    prev_x, prev_y = 0, 0
    draw_points = deque(maxlen=512)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning(" Cannot access webcam.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        h, w, _ = frame.shape
        index_tip = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get index finger tip position
                index_tip = hand_landmarks.landmark[8]
                x, y = int(index_tip.x * w), int(index_tip.y * h)

                # Get thumb tip position to check if “pinched” (stop writing)
                thumb_tip = hand_landmarks.landmark[4]
                thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
                distance = np.sqrt((x - thumb_x) ** 2 + (y - thumb_y) ** 2)

                if distance > 40:  # Writing mode
                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = x, y
                    if eraser_mode:
                        cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 0), 40)
                    else:
                        color_bgr = tuple(int(brush_color.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
                        cv2.line(canvas, (prev_x, prev_y), (x, y), color_bgr, brush_size)
                    prev_x, prev_y = x, y
                else:
                    prev_x, prev_y = 0, 0  # Stop writing

        # Combine camera feed and canvas
        frame = cv2.addWeighted(frame, 0.6, canvas, 0.4, 0)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB")

    cap.release()
    st.success("Webcam stopped.")
else:
    st.warning(" Enable webcam from the sidebar to start tracking.")
