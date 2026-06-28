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

```
model = RFDETRNano()

try:
    import torch
    model.optimize_for_inference(
        dtype=torch.float16
    )
except:
    pass

return model
```

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

```
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    image_np = np.array(image)

    st.image(
        image,
        caption="Uploaded Image",
        width="stretch"
    )

    with st.spinner(
        "Detecting Objects..."
    ):

        detections = model.predict(
            image_np
        )

        box_annotator = sv.BoxAnnotator()

        labels = []

        if (
            hasattr(detections, "data")
            and "class_name"
            in detections.data
        ):
            labels = [
                f"{cls}"
                for cls in detections.data[
                    "class_name"
                ]
            ]

        annotated_image = (
            box_annotator.annotate(
                scene=image_np.copy(),
                detections=detections
            )
        )

    st.image(
        annotated_image,
        caption="Detection Result",
        width="stretch"
    )

    total_objects = len(
        detections.xyxy
    )

    st.success(
        f"🎯 Total Objects Detected: {total_objects}"
    )

    object_counts = {}

    if (
        hasattr(detections, "data")
        and "class_name"
        in detections.data
    ):

        for cls in detections.data[
            "class_name"
        ]:

            object_counts[cls] = (
                object_counts.get(
                    cls, 0
                ) + 1
            )

    if object_counts:

        st.subheader(
            "📊 Object Summary"
        )

        for obj, count in sorted(
            object_counts.items()
        ):
            st.write(
                f"✅ {obj}: {count}"
            )
```

# ==================================================

# VIDEO DETECTION

# ==================================================

else:

```
st.warning(
    "For best performance upload videos under 30 MB."
)

uploaded_video = st.file_uploader(
    "Upload Video",
    type=["mp4", "avi", "mov"]
)

if uploaded_video:

    size_mb = (
        uploaded_video.size
        / (1024 * 1024)
    )

    st.info(
        f"Video Size: {size_mb:.1f} MB"
    )

    if size_mb > 50:

        st.error(
            "⚠ Please upload video below 50 MB."
        )

        st.stop()

    st.success(
        "✅ Video uploaded successfully"
    )

    if st.button(
        "▶ Start Detection"
    ):

        with st.spinner(
            "Processing video..."
        ):

            temp_video = (
                tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                )
            )

            temp_video.write(
                uploaded_video.read()
            )

            temp_video.close()

            cap = cv2.VideoCapture(
                temp_video.name
            )

            frame_placeholder = (
                st.empty()
            )

            progress_bar = (
                st.progress(0)
            )

            total_frames = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            frame_count = 0
            processed_frames = 0

            total_detected = {}

            box_annotator = (
                sv.BoxAnnotator()
            )

            while cap.isOpened():

                ret, frame = cap.read()

                if not ret:
                    break

                frame_count += 1

                progress_bar.progress(
                    min(
                        frame_count
                        / max(
                            total_frames,
                            1
                        ),
                        1.0
                    )
                )

                # Fast mode
                if frame_count % 120 != 0:
                    continue

                processed_frames += 1

                frame = cv2.resize(
                    frame,
                    (640, 360)
                )

                frame_rgb = (
                    cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )
                )

                detections = (
                    model.predict(
                        frame_rgb
                    )
                )

                if (
                    hasattr(
                        detections,
                        "data"
                    )
                    and "class_name"
                    in detections.data
                ):

                    for cls in detections.data[
                        "class_name"
                    ]:

                        total_detected[
                            cls
                        ] = (
                            total_detected.get(
                                cls,
                                0
                            )
                            \+ 1
                        )

                annotated_frame = (
                    box_annotator.annotate(
                        scene=frame_rgb.copy(),
                        detections=detections
                    )
                )

                frame_placeholder.image(
                    annotated_frame,
                    width="stretch"
                )

            cap.release()

        st.success(
            "✅ Video Detection Completed"
        )

        st.subheader(
            "📊 Objects Found In Video"
        )

        if not total_detected:

            st.warning(
                "No objects detected."
            )

        else:

            total_objects = sum(
                total_detected.values()
            )

            st.success(
                f"🎯 Total Objects: {total_objects}"
            )

            for obj, count in sorted(
                total_detected.items()
            ):
                st.write(
                    f"✅ {obj}: {count}"
                )

            st.success(
                f"Processed Frames: {processed_frames}"
            )
```
