import streamlit as st
import numpy as np
from PIL import Image
from rfdetr import RFDETRBase
import supervision as sv
import cv2
import tempfile

st.title("RF-DETR Object Detection")

option = st.radio(
    "Choose Input Type",
    ["Image", "Video"]
)

model = RFDETRBase()

# ---------------- IMAGE ----------------

if option == "Image":

    uploaded_image = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:

        image = Image.open(uploaded_image).convert("RGB")
        image_np = np.array(image)

        detections = model.predict(image_np)

        box_annotator = sv.BoxAnnotator()

        annotated_image = box_annotator.annotate(
            scene=image_np.copy(),
            detections=detections
        )

        st.image(
            annotated_image,
            caption="Detected Objects",
            use_container_width=True
        )

# ---------------- VIDEO ----------------

if option == "Video":

    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video:

        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_video.read())

        cap = cv2.VideoCapture(temp_file.name)

        stframe = st.empty()

        box_annotator = sv.BoxAnnotator()


    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        if frame_count % 10 != 0:
            continue

        detections = model.predict(frame)

        annotated_frame = box_annotator.annotate(
            scene=frame.copy(),
            detections=detections
    )

    stframe.image(
        annotated_frame,
        channels="BGR",
        use_container_width=True
    )
        cap.release()
