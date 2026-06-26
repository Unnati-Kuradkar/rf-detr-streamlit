import streamlit as st
import numpy as np
from PIL import Image
from rfdetr import RFDETRBase
import supervision as sv

st.title("RF-DETR Object Detection")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image, caption="Uploaded Image")

    st.write("Loading model...")
    model = RFDETRBase()

    st.write("Detecting objects...")
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
        use_container_width=True
    )

    st.subheader("Detected Objects")

    for name, conf in zip(
        detections.data["class_name"],
        detections.confidence
    ):
        st.write(f"✅ {name} ({conf:.2f})")
