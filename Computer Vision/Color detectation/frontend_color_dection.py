import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile

st.set_page_config(page_title="Color Detection", layout="wide")

st.title(" Real-Time Color Detection")
st.markdown("Use your webcam to detect specific colors in real-time.")

# Sidebar controls
st.sidebar.header(" Controls")
run = st.sidebar.checkbox("Start Camera", value=False)
selected_color = st.sidebar.selectbox(
    "Select Color to Detect",
    ["Red", "Green", "Blue", "Custom"]
)
st.sidebar.markdown("---")

# Define HSV ranges for predefined colors
color_ranges = {
    "Red": ([0, 120, 70], [10, 255, 255]),
    "Green": ([36, 100, 100], [86, 255, 255]),
    "Blue": ([94, 80, 2], [126, 255, 255])
}

# Custom color range
if selected_color == "Custom":
    h_min = st.sidebar.slider("Hue Min", 0, 179, 0)
    s_min = st.sidebar.slider("Sat Min", 0, 255, 100)
    v_min = st.sidebar.slider("Val Min", 0, 255, 100)
    h_max = st.sidebar.slider("Hue Max", 0, 179, 179)
    s_max = st.sidebar.slider("Sat Max", 0, 255, 255)
    v_max = st.sidebar.slider("Val Max", 0, 255, 255)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
else:
    lower = np.array(color_ranges[selected_color][0])
    upper = np.array(color_ranges[selected_color][1])

# Frame display area
FRAME_WINDOW = st.image([])

# Start camera logic
if run:
    cap = cv2.VideoCapture(0)
    st.sidebar.success(" Camera started. Uncheck to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame from camera.")
            break

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Mask for selected color
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Combine images: original | mask | result
        stacked = np.hstack((frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), result))
        img_rgb = cv2.cvtColor(stacked, cv2.COLOR_BGR2RGB)

        # Display in Streamlit
        FRAME_WINDOW.image(img_rgb)

        # Stop camera when checkbox is unchecked
        if not st.session_state.get("Start Camera", True):
            break

    cap.release()
else:
    st.warning("▶ Click **Start Camera** to begin color detection.")

# Snapshot button
st.sidebar.markdown("---")
if st.sidebar.button("Capture Snapshot"):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if ret:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(img, caption="Captured Image")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            Image.fromarray(img).save(tmp.name)
            st.success(f"💾 Snapshot saved at: {tmp.name}")
    else:
        st.error("❌ Failed to capture image.")

