# File: frontend_video_frame.py
import streamlit as st
import cv2
import tempfile

st.title("  Video Frame Processing with OpenCV") 

# Choose input type
use_webcam = st.checkbox("Use Webcam")
uploaded_file = st.file_uploader("Or Upload a Video", type=["mp4", "avi", "mov"])

# Choose processing mode
mode = st.selectbox("Select Processing Mode", ["Original", "Grayscale", "Edge Detection", "Blurred"])

stframe = st.empty()

def process_frame(frame, mode):
    """Apply selected processing effect."""
    if mode == "Grayscale":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif mode == "Edge Detection":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, 100, 200)
    elif mode == "Blurred":
        return cv2.GaussianBlur(frame, (15, 15), 0)
    else:
        return frame

def run_video(video_source):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        st.warning("Cannot open video source.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror for webcam
        processed = process_frame(frame, mode)

        # If grayscale or edges, convert single channel to 3 for Streamlit
        if len(processed.shape) == 2:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        else:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

        stframe.image(processed, channels="RGB", use_container_width=True)

    cap.release()

# Handle input source
if use_webcam:
    run_video(0)
elif uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    run_video(tfile.name)
