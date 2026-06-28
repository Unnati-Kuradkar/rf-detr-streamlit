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

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        image_np = np.array(image)

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
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
            caption="Detection Result",
            use_container_width=True
        )

        total_objects = len(detections.xyxy)

        st.success(
            f"🎯 Total Objects Detected: {total_objects}"
        )

        object_counts = {}

        if (
            hasattr(detections, "data")
            and "class_name" in detections.data
        ):

            for cls in detections.data["class_name"]:

                object_counts[cls] = (
                    object_counts.get(cls, 0) + 1
                )

        if object_counts:

            st.subheader("📊 Object Summary")

            for obj, count in sorted(
                object_counts.items()
            ):
                st.write(
                    f"✅ {obj}: {count}"
                )

# =================================================
