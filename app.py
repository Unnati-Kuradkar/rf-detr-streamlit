import streamlit as st
import numpy as np
from PIL import Image
from rfdetr import RFDETRNano
import supervision as sv

st.set_page_config(page_title="Object Detection", layout="wide")

st.title("🚗 RF-DETR Object Detection")

# Load model only once
@st.cache_resource
def load_model():
    model = RFDETRNano()
    return model

model = load_model()

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image, caption="Uploaded Image", width="stretch")

    with st.spinner("Detecting Objects..."):

        detections = model.predict(image_np)

        # Draw boxes
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

    # ==========================
    # OBJECT COUNT
    # ==========================
    total_objects = len(detections)

    st.success(f"🎯 Total Objects Detected: {total_objects}")

    # ==========================
    # OBJECT WISE COUNT
    # ==========================
    object_counts = {}

    if "class_name" in detections.data:

        for cls in detections.data["class_name"]:
            object_counts[cls] = object_counts.get(cls, 0) + 1

        st.subheader("📊 Object Summary")

        for obj, count in object_counts.items():
            st.write(f"✅ {obj}: {count}")

    # ==========================
    # DETAILED LIST
    # ==========================
    st.subheader("📋 Detection Details")

    if "class_name" in detections.data:

        for name, conf in zip(
            detections.data["class_name"],
            detections.confidence
        ):
            st.write(
                f"🔹 {name} ({conf:.2f})"
            )
