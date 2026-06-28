import streamlit as st
import numpy as np
from PIL import Image
from rfdetr import RFDETRNano
import supervision as sv

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="RF-DETR Object Detection",
    layout="wide"
)

st.title("🚗 RF-DETR Object Detection")

# ----------------------------
# LOAD MODEL ONLY ONCE
# ----------------------------
@st.cache_resource
def load_model():
    return RFDETRNano()

model = load_model()

# ----------------------------
# IMAGE UPLOAD
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

# ----------------------------
# PROCESS IMAGE
# ----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.subheader("📸 Uploaded Image")
    st.image(image)

    with st.spinner("🔍 Detecting Objects..."):

        detections = model.predict(image_np)

        box_annotator = sv.BoxAnnotator()

        annotated_image = box_annotator.annotate(
            scene=image_np.copy(),
            detections=detections
        )

    # ----------------------------
    # DETECTED IMAGE
    # ----------------------------
    st.subheader("🎯 Detection Result")
    st.image(annotated_image)

    # ----------------------------
    # TOTAL OBJECT COUNT
    # ----------------------------
    total_objects = len(detections.xyxy)

    st.success(
        f"🎯 Total Objects Detected: {total_objects}"
    )

    # ----------------------------
    # OBJECT COUNTS
    # ----------------------------
    object_counts = {}

    if (
        hasattr(detections, "data")
        and "class_name" in detections.data
    ):

        for cls in detections.data["class_name"]:
            object_counts[cls] = (
                object_counts.get(cls, 0) + 1
            )

        st.subheader("📊 Object Summary")

        for obj, count in sorted(object_counts.items()):
            st.write(
                f"✅ **{obj}** : {count}"
            )

    # ----------------------------
    # DETAILED DETECTIONS
    # ----------------------------
    st.subheader("📋 Detection Details")

    if (
        hasattr(detections, "data")
        and "class_name" in detections.data
    ):

        for name, conf in zip(
            detections.data["class_name"],
            detections.confidence
        ):

            st.write(
                f"🔹 {name} ({conf:.2f})"
            )
