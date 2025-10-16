# File: frontend_virtual_math.py
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.title(" Virtual Hand Gesture Math with MediaPipe")

use_webcam = st.checkbox("Use Webcam for Gesture Control")

if use_webcam:
    stframe = st.empty()

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

    cap = cv2.VideoCapture(0)

    st.markdown("""
    ###  Instructions:
    - **1 finger** → Addition  
    - **2 fingers** → Subtraction  
    - **3 fingers** → Multiplication  
    - **4 fingers** → Division  
    - **5 fingers** → Reset  
    """)

    num1, num2 = 8, 4  # You can change these numbers
    result = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Cannot access webcam.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        frame_h, frame_w, _ = frame.shape
        gesture_text = "Waiting..."

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get finger tip positions
                landmarks = hand_landmarks.landmark
                finger_tips = [8, 12, 16, 20]
                open_fingers = 0

                for tip in finger_tips:
                    if landmarks[tip].y < landmarks[tip - 2].y:
                        open_fingers += 1

                # Thumb detection (optional)
                thumb_tip = landmarks[4]
                thumb_base = landmarks[2]
                if thumb_tip.x > thumb_base.x:
                    open_fingers += 1

                # Detect gesture and calculate math operation
                if open_fingers == 1:
                    result = num1 + num2
                    gesture_text = f"Addition: {num1} + {num2} = {result}"
                elif open_fingers == 2:
                    result = num1 - num2
                    gesture_text = f"Subtraction: {num1} - {num2} = {result}"
                elif open_fingers == 3:
                    result = num1 * num2
                    gesture_text = f"Multiplication: {num1} × {num2} = {result}"
                elif open_fingers == 4:
                    result = round(num1 / num2, 2)
                    gesture_text = f"Division: {num1} ÷ {num2} = {result}"
                elif open_fingers == 5:
                    result = None
                    gesture_text = "Reset"

        # Overlay gesture text on video
        cv2.putText(frame, gesture_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Display frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

    cap.release()
    st.success("Webcam stopped.")
