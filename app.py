import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tempfile

from rfdetr import RFDETRBase
import supervision as sv

st.set_page_config(
    page_title="RF-DETR Object Detection",
    layout="wide"
)

st.title("🚗 RF-DETR Object Detection")

# ------------------------------
# Load model only once
# ------------------------------
@st.cache_resource
def load_model():
    model = RFDETRBase()
    model.optimize_for_inference()
    return model

model = load_model()

option = st.radio(
    "Choose Input Type",
    ["Image", "Video"]
)

# =====================================================
# IMAGE DETECTION
# =====================================================
if option == "Image":

    uploaded_image = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:

        image = Image.open(uploaded_image).convert("RGB")
        image_np = np.array(image)

        st.image(
            image,
            caption="Uploaded Image",
            width="stretch"
        )

        with st.spinner("Detecting objects..."):

            detections = model.predict(image_np)

            box_annotator = sv.BoxAnnotator()

            annotated_image = box_annotator.annotate(
                scene=image_np.copy(),
                detections=detections
            )

        st.image(
            annotated_image,
            caption="Detected Objects",
            width="stretch"
        )

        st.subheader("Detected Objects")

        for name, conf in zip(
            detections.data["class_name"],
            detections.confidence
        ):
            st.write(f"✅ {name} ({conf:.2f})")

# =====================================================
# VIDEO DETECTION
# =====================================================
else:

    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video:

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)

        frame_placeholder = st.empty()

        st.warning(
            "Processing every 10th frame for faster speed."
        )

        frame_count = 0

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            # Skip frames
            if frame_count % 10 != 0:
                continue

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            detections = model.predict(frame_rgb)

            box_annotator = sv.BoxAnnotator()

            annotated = box_annotator.annotate(
                scene=frame_rgb.copy(),
                detections=detections
            )

            frame_placeholder.image(
                annotated,
                width="stretch"
            )

        cap.release()

        st.success("Video Processing Completed!")
