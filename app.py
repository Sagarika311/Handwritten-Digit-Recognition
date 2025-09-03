import streamlit as st 
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
from streamlit_drawable_canvas import st_canvas

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Digit Recognition",
    page_icon="🖼️",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🖼️ Handwritten Digit Recognition")
st.write("Upload images, use a sample image, or draw a digit, and the model will predict it.")

# ---------------------------
# Sidebar Instructions
# ---------------------------
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Upload one or multiple images of handwritten digits.
2. Each image should contain a **clear single digit**.
3. Supported file types: **PNG, JPG, JPEG**.
4. Use the sample image or draw a digit for testing.
""")

# ---------------------------
# Reload Button
# ---------------------------
if "reload_trigger" not in st.session_state:
    st.session_state.reload_trigger = 0
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "drawn_image" not in st.session_state:
    st.session_state.drawn_image = None
if "draw_mode" not in st.session_state:
    st.session_state.draw_mode = False

def reload_app():
    st.session_state.reload_trigger += 1
    st.session_state.uploaded_files = []
    st.session_state.drawn_image = None
    st.session_state.draw_mode = False

st.sidebar.button("Reload App", on_click=reload_app)
st.write(f"App reloaded {st.session_state.reload_trigger} times")

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_digit_model():
    return load_model("digit_recognition_model.h5")

try:
    model = load_digit_model()
except Exception as e:
    st.error(f"Failed to load model. Make sure 'digit_recognition_model.h5' is in the same folder.\nError: {e}")
    st.stop()

# ---------------------------
# Image Processing Function
# ---------------------------
def process_image(img):
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = 255 - img_array  # invert if background is black
    img_array = img_array / 255.0
    return img_array.reshape(1, 28, 28, 1)

# ---------------------------
# Sample Image & Upload
# ---------------------------
sample_img_path = "sample_digit.png"
use_sample = st.sidebar.button("Use Sample Image") and os.path.exists(sample_img_path)

if use_sample:
    st.session_state.uploaded_files = [sample_img_path]
elif not st.session_state.uploaded_files:
    st.session_state.uploaded_files = st.file_uploader(
        "Choose image(s)...", type=["png", "jpg", "jpeg"], accept_multiple_files=True
    )

# ---------------------------
# Draw Digit (Persistent)
# ---------------------------
if st.sidebar.button("Draw Digit"):
    st.session_state.draw_mode = not st.session_state.draw_mode  # toggle draw mode

if st.session_state.draw_mode:
    st.subheader("Draw a Digit Below 👇")
    col1, col2 = st.columns([1, 1])  # canvas and prediction side by side

    with col1:
        canvas_result = st_canvas(
            fill_color="#000000",
            stroke_width=10,
            stroke_color="#FFFFFF",
            background_color="#000000",
            width=280,
            height=280,
            drawing_mode="freedraw",
            key="canvas"
        )

    # Always show Loading placeholder first
    pred_placeholder = col2.empty()
    pred_placeholder.info("Loading...")  

    if canvas_result.image_data is not None:
        drawn_img = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("L")
        st.session_state.drawn_image = drawn_img

        # Predict
        img_array = process_image(drawn_img)
        prediction = model.predict(img_array)
        digit = np.argmax(prediction)

        # Display prediction and drawn image
        pred_placeholder.success(f"Predicted Digit: **{digit}**")
        col2.image(drawn_img.resize((150, 150)), caption="Drawn Digit", use_container_width=False)

# ---------------------------
# Predictions for Uploaded / Sample Images
# ---------------------------
all_images = []

for file in st.session_state.uploaded_files:
    try:
        if isinstance(file, str):
            img = Image.open(file)
        else:
            img = Image.open(file)
        all_images.append(("Uploaded Image", img))
    except Exception as e:
        st.error(f"Error loading image {file}: {e}")

if all_images:
    st.subheader("Predictions for Uploaded / Sample Images:")
    for title, img in all_images:
        img_placeholder = st.empty()
        pred_placeholder = st.empty()
        img_placeholder.image(img.resize((150, 150)), caption=title, use_container_width=False)
        pred_placeholder.info("Loading...")  # default Loading

        img_array = process_image(img)
        prediction = model.predict(img_array)
        digit = np.argmax(prediction)
        pred_placeholder.success(f"Predicted Digit: **{digit}**")
else:
    if not st.session_state.draw_mode:
        st.info("Upload images, use a sample image, or draw a digit to start predicting.")
