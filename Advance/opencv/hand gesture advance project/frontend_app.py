# File: frontend_hand_cursor.py
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.title("🖱️ Hand Cursor Tracking with MediaPipe")

use_webcam = st.checkbox("Use Webcam for Hand Cursor")

if use_webcam:
    stframe = st.empty()

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

    cap = cv2.VideoCapture(0)
    screen_w, screen_h = 1280, 720  # Virtual screen dimensions

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Cannot access webcam.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        frame_h, frame_w, _ = frame.shape

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get index finger tip and thumb tip
                index_tip = hand_landmarks.landmark[8]
                thumb_tip = hand_landmarks.landmark[4]

                # Convert normalized coordinates to pixel positions
                x = int(index_tip.x * frame_w)
                y = int(index_tip.y * frame_h)
                thumb_x = int(thumb_tip.x * frame_w)
                thumb_y = int(thumb_tip.y * frame_h)

                # Draw cursor
                cv2.circle(frame, (x, y), 15, (0, 255, 255), cv2.FILLED)

                # Calculate distance between thumb and index finger
                distance = np.hypot(x - thumb_x, y - thumb_y)

                # Detect pinch (click) gesture
                if distance < 40:
                    cv2.circle(frame, (x, y), 25, (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, "Click!", (x + 20, y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display the result
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

    cap.release()
    st.success("Webcam stopped.")
