# File: frontend_handgame.py
import streamlit as st
import cv2
import mediapipe as mp
import random
import time

st.title("🎮 Hand Gesture Game – Rock, Paper, Scissors")

# Button to start game
start_game = st.button("Start Game")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Gesture names
gestures = ["Rock", "Paper", "Scissors"]

def detect_gesture(hand_landmarks):
    """
    Basic gesture detection based on finger positions.
    Returns 'Rock', 'Paper', or 'Scissors'
    """
    landmarks = hand_landmarks.landmark

    # Finger tip landmarks
    finger_tips = [8, 12, 16, 20]
    thumb_tip = 4

    # Count how many fingers are open
    open_fingers = 0
    for tip in finger_tips:
        if landmarks[tip].y < landmarks[tip - 2].y:
            open_fingers += 1

    # Thumb logic
    if landmarks[thumb_tip].x > landmarks[thumb_tip - 1].x:
        open_fingers += 1

    # Decide gesture
    if open_fingers == 0:
        return "Rock"
    elif open_fingers == 5:
        return "Paper"
    elif open_fingers in [2, 3]:
        return "Scissors"
    else:
        return "Unknown"

# Webcam stream
if start_game:
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    st.write("✋ Make your move!")

    computer_choice = random.choice(gestures)
    start_time = time.time()
    countdown_duration = 5  # seconds

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Cannot access webcam.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        user_gesture = "Waiting..."

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                user_gesture = detect_gesture(hand_landmarks)

        elapsed = int(time.time() - start_time)
        remaining = countdown_duration - elapsed

        if remaining > 0:
            cv2.putText(frame, f"Show your move in {remaining}s", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        else:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

    cap.release()

    # Decide winner
    st.subheader("🧠 Results")
    st.write(f"**Your Move:** {user_gesture}")
    st.write(f"**Computer's Move:** {computer_choice}")

    if user_gesture == computer_choice:
        st.success("🤝 It's a Tie!")
    elif (user_gesture == "Rock" and computer_choice == "Scissors") or \
         (user_gesture == "Paper" and computer_choice == "Rock") or \
         (user_gesture == "Scissors" and computer_choice == "Paper"):
        st.success("🎉 You Win!")
    else:
        st.error("💻 Computer Wins!")
