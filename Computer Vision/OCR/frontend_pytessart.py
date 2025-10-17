import streamlit as st
from PIL import Image
import pytesseract
import io

#  If Tesseract is not in PATH, specify the correct path here:
#  Make sure this matches your installed Tesseract location.
# If you installed it in a different folder, update the path below.
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Streamlit page configuration
st.set_page_config(page_title=" OCR Text Extraction", layout="wide")

# App title and description
st.title(" Optical Character Recognition (OCR) App")
st.markdown("Upload an image and extract text using **Tesseract OCR**.")

# Sidebar for file upload and settings
st.sidebar.header(" Upload Section")
uploaded_file = st.sidebar.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
st.sidebar.markdown("---")
lang = st.sidebar.text_input("Language (default = 'eng')", "eng")

# Main content area
if uploaded_file:
    # Open and display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.info("Processing image for text extraction...")

    try:
        # Perform OCR
        extracted_text = pytesseract.image_to_string(image, lang=lang)

        # Display results
        st.success(" Text successfully extracted!")
        st.subheader("Extracted Text:")
        st.text_area("Recognized Text", extracted_text, height=250)

        # Download option
        st.download_button(
            label=" Download as .txt",
            data=extracted_text,
            file_name="extracted_text.txt",
            mime="text/plain"
        )

    except pytesseract.TesseractNotFoundError:
        st.error(" Tesseract not found! Please check the installation path above.")
    except Exception as e:
        st.error(f" An error occurred: {e}")

else:
    st.warning(" Upload an image from the sidebar to start OCR processing.")

# Footer
st.markdown("---")
st.caption("Built with  Tesseract OCR + Streamlit")

