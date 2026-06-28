import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tempfile
import supervision as sv
from rfdetr import RFDETRNano
import torch

@st.cache_resource
def load_model():
    model = RFDETRNano()

    try:
        model.optimize_for_inference(
            dtype=torch.float16
        )
    except:
        pass

    return model

model = load_model()

st.title("🚗 RF-DETR Object Detection")

# =========================
# LOAD MODEL ONCE
# =========================
@st.cache_resource
def load_model():
    model = RFDETRNano()
    return model

model = load_model()

option = st.radio(
    "Choose Input Type",
    ["Image", "Video"]
)

# ==================================================
# IMAGE DETECTION
# ==================================================
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

        with st.spinner("Detecting Objects..."):

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

# ==================================================
# VIDEO DETECTION
# ==================================================
else:

    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video:

        st.video(uploaded_video)

        if st.button("Start Detection"):

            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            temp_file.write(uploaded_video.read())
            temp_file.close()

            cap = cv2.VideoCapture(temp_file.name)

            frame_placeholder = st.empty()
            progress_bar = st.progress(0)

            frame_count = 0
            processed_frames = 0
            MAX_FRAMES = 100

            box_annotator = sv.BoxAnnotator()

            st.info(
                "Fast Mode: Processing every 60th frame"
            )

            while cap.isOpened():

                ret, frame = cap.read()

                if not ret:
                    break

                frame_count += 1

                # Skip most frames
                if frame_count % 60 != 0:
                    continue

                processed_frames += 1

                if processed_frames > MAX_FRAMES:
                    break

                frame_rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                # Resize for speed
                frame_rgb = cv2.resize(
                    frame_rgb,
                    (640, 360)
                )

                detections = model.predict(
                    frame_rgb
                )

                annotated = box_annotator.annotate(
                    scene=frame_rgb.copy(),
                    detections=detections
                )

                frame_placeholder.image(
                    annotated,
                    width="stretch"
                )

                progress_bar.progress(
                    min(
                        processed_frames / MAX_FRAMES,
                        1.0
                    )
                )

            cap.release()

            st.success(
                f"Detection Complete! Processed {processed_frames} frames."
            )
