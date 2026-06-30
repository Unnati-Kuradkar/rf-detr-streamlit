import streamlit as st
import numpy as np
from PIL import Image
from rfdetr import RFDETRNano
import supervision as sv
from supervision import ByteTrack
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

    model = RFDETRNano()

    try:
        import torch
        model.optimize_for_inference(
            dtype=torch.float16
        )
    except:
        pass

    return model


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

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        with st.spinner("Detecting Objects..."):

            detections = model.predict(image_np)

            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator()

            labels = []

            if (
                hasattr(detections, "data")
                and "class_name" in detections.data
            ):

                labels = [
                    f"{cls} {conf:.2f}"
                    for cls, conf in zip(
                        detections.data["class_name"],
                        detections.confidence
                    )
                ]

            annotated_image = box_annotator.annotate(
                scene=image_np.copy(),
                detections=detections
            )

            annotated_image = label_annotator.annotate(
                scene=annotated_image,
                detections=detections,
                labels=labels
            )

        st.image(
            annotated_image,
            caption="Detection Result",
            use_container_width=True
        )

        st.success(
            f"🎯 Total Objects Detected: {len(detections.xyxy)}"
        )

        st.subheader("📋 Detection Details")

        if (
            hasattr(detections, "data")
            and "class_name" in detections.data
        ):

            for i, box in enumerate(detections.xyxy):

                x1, y1, x2, y2 = box

                width = int(x2 - x1)
                height = int(y2 - y1)

                name = detections.data["class_name"][i]
                conf = detections.confidence[i]

                st.write(
                    f"🔹 {name} | Confidence: {conf:.2f} | Width: {width}px | Height: {height}px"
                )

        object_counts = {}

        if (
            hasattr(detections, "data")
            and "class_name" in detections.data
        ):

            for cls in detections.data["class_name"]:
                object_counts[cls] = object_counts.get(cls, 0) + 1

        if object_counts:

            st.subheader("📊 Object Summary")

            for obj, count in sorted(object_counts.items()):
                st.write(f"✅ {obj}: {count}")
# =================================================
# ==================================================
# VIDEO DETECTION
# ==================================================

if input_type == "Video":

    st.success("🎥 VIDEO MODE ACTIVE")

    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"],
        key="video_uploader"
    )

    if uploaded_video is not None:

        st.success("✅ Video uploaded successfully")

        st.video(uploaded_video)

        st.write(
            "Video Size:",
            round(uploaded_video.size / (1024 * 1024), 2),
            "MB"
        )

        if st.button(
            "▶ Start Detection",
            key="start_detection_btn"
        ):

            st.success("🚀 Detection Started")

            with st.spinner("Processing Video..."):

                temp_video = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                )

                temp_video.write(
                    uploaded_video.read()
                )

                temp_video.close()

                cap = cv2.VideoCapture(
                    temp_video.name
                )

                frame_placeholder = st.empty()

               frame_count = 0
               processed_frames = 0

               unique_objects = {}

               tracker = ByteTrack()

               box_annotator = sv.BoxAnnotator()
               label_annotator = sv.LabelAnnotator()

                while cap.isOpened():

                    ret, frame = cap.read()

                    if not ret:
                        break

                    frame_count += 1

                    # Fast Processing
                    if frame_count % 10 != 0:
                        continue

                    processed_frames += 1

                    frame = cv2.resize(
                        frame,
                        (640, 360)
                    )

                    frame_rgb = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )

                    detections = model.predict(
                        frame_rgb
                    )
                    detections = tracker.update_with_detections(
                        detections
                    )

                    labels = []

                    if (
                        hasattr(detections, "data")
                        and "class_name" in detections.data
                    ):

                        labels = [
                            f"{cls} {conf:.2f}"
                            for cls, conf in zip(
                                detections.data["class_name"],
                                detections.confidence
                            )
                        ]

                        for cls, track_id in zip(
                            detections.data["class_name"],
                            detections.tracker_id
                        ):

                        if track_id is None:
                            continue

                        if cls not in unique_objects:
                            unique_objects[cls] = set()

                        unique_objects[cls].add(
                            int(track_id)
                        )

                    annotated_frame = box_annotator.annotate(
                        scene=frame_rgb.copy(),
                        detections=detections
                    )

                    annotated_frame = label_annotator.annotate(
                        scene=annotated_frame,
                        detections=detections,
                        labels=labels
                    )

                    frame_placeholder.image(
                        annotated_frame,
                        use_container_width=True
                    )

                cap.release()

            st.success(
                f"✅ Video Detection Completed | Processed Frames: {processed_frames}"
            )

            st.subheader(
                "📊 Objects Found In Video"
            )

            if len(unique_objects) == 0:
                st.warning(
                "No Objects Detected"
            )

            else:

            total_unique = 0

            for ids in unique_objects.values():
                total_unique += len(ids)

            st.success(
                f"🎯 Total Unique Objects: {total_unique}"
            )

            for obj, ids in sorted(
                unique_objects.items()
            ):

            st.write(
                f"✅ {obj}: {len(ids)}"
            )
