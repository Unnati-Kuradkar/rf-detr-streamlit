import streamlit as st
import numpy as np
from PIL import Image
from rfdetr import RFDETRNano
import supervision as sv
import cv2
import tempfile

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(
    page_title="RF-DETR Object Detection",
    layout="wide"
)

st.title("🚗 RF-DETR Object Detection")

# --------------------------------
# LOAD MODEL ONCE
# --------------------------------
@st.cache_resource
def load_model():
    return RFDETRNano()

model = load_model()

# --------------------------------
# INPUT TYPE
# --------------------------------
input_type = st.radio(
    "Choose Input Type",
    ["Image", "Video"]
)

# ==================================================
# IMAGE DETECTION
# ==================================================
if input_type == "Image":

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        st.image(image, caption="Uploaded Image")

        with st.spinner("Detecting Objects..."):

            detections = model.predict(image_np)

            box_annotator = sv.BoxAnnotator()

            annotated_image = box_annotator.annotate(
                scene=image_np.copy(),
                detections=detections
            )

        st.image(
            annotated_image,
            caption="Detection Result"
        )

        total_objects = len(detections.xyxy)

        st.success(
            f"🎯 Total Objects Detected: {total_objects}"
        )

        object_counts = {}

        if "class_name" in detections.data:

            for cls in detections.data["class_name"]:
                object_counts[cls] = (
                    object_counts.get(cls, 0) + 1
                )

            st.subheader("📊 Object Summary")

            for obj, count in object_counts.items():
                st.write(f"✅ {obj}: {count}")

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

        if st.button("▶ Start Detection"):

            with st.spinner(
                "Processing video... Please wait"
            ):

                temp_video = tempfile.NamedTemporaryFile(
                    delete=False
                )

                temp_video.write(
                    uploaded_video.read()
                )

                cap = cv2.VideoCapture(
                    temp_video.name
                )

                frame_placeholder = st.empty()

                frame_count = 0
                processed_frames = 0

                total_detected = {}

                box_annotator = sv.BoxAnnotator()

                while cap.isOpened():

                    ret, frame = cap.read()

                    if not ret:
                        break

                    frame_count += 1

                    # Fast mode
                    if frame_count % 60 != 0:
                        continue

                    processed_frames += 1

                    frame_rgb = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )

                    detections = model.predict(
                        frame_rgb
                    )

                    if (
                        hasattr(detections, "data")
                        and "class_name"
                        in detections.data
                    ):

                        for cls in detections.data[
                            "class_name"
                        ]:

                            total_detected[cls] = (
                                total_detected.get(
                                    cls, 0
                                )
                                + 1
                            )

                    annotated_frame = (
                        box_annotator.annotate(
                            scene=frame_rgb.copy(),
                            detections=detections
                        )
                    )

                    frame_placeholder.image(
                        annotated_frame
                    )

                cap.release()

            st.success(
                "✅ Video Detection Completed"
            )

            st.subheader(
                "📊 Objects Found In Video"
            )

            if len(total_detected) == 0:

                st.warning(
                    "No objects detected."
                )

            else:

                for obj, count in sorted(
                    total_detected.items()
                ):
                    st.write(
                        f"✅ {obj}: {count}"
                    )

                st.success(
                    f"Processed Frames: {processed_frames}"
                )
