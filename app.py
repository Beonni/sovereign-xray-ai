
import streamlit as st
from PIL import Image
import numpy as np
import hashlib
import torch

st.set_page_config(
    page_title="Sovereign X-ray AI",
    page_icon="🏥",
    layout="wide"
)

MODEL_ID = "densenet121-res224-all"

@st.cache_resource(show_spinner="Loading X-ray AI model...")
def load_model():
    import torchxrayvision as xrv
    model = xrv.models.DenseNet(weights=MODEL_ID)
    model.eval()
    return model

def predict(image):
    import torchxrayvision as xrv

    img = np.asarray(
        image.convert("L"), dtype=np.float32
    )
    img = xrv.datasets.normalize(img, 255)
    img = img[None, :, :]

    img = xrv.datasets.XRayCenterCrop()(img)
    img = xrv.datasets.XRayResizer(224)(img)

    tensor = torch.from_numpy(
        np.ascontiguousarray(img)
    ).unsqueeze(0).float()

    model = load_model()

    with torch.inference_mode():
        outputs = model(tensor)[0].cpu().numpy()

    return sorted(
        [
            (name, float(score))
            for name, score in
            zip(model.pathologies, outputs)
            if np.isfinite(score)
        ],
        key=lambda item: item[1],
        reverse=True
    )

if "xray_image" not in st.session_state:
    st.session_state.xray_image = None

if "image_hash" not in st.session_state:
    st.session_state.image_hash = None

if "results" not in st.session_state:
    st.session_state.results = None

st.title("🏥 Sovereign AI Medical Imaging System")
st.caption("基於主權 AI 之胸部 X 光影像輔助分析系統")
st.warning(
    "Educational research only. "
    "Not validated for clinical diagnosis."
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Upload X-ray",
        "AI Analysis",
        "Doctor Review",
        "Model Control",
        "Audit Log"
    ]
)

if page == "Dashboard":
    st.header("System Dashboard")
    st.success("Website is running!")
    st.write("Inference environment: Cloud CPU")
    st.write("AI model:", MODEL_ID)

elif page == "Upload X-ray":
    st.header("Upload Chest X-ray")

    file = st.file_uploader(
        "Upload a de-identified X-ray",
        type=["png", "jpg", "jpeg"]
    )

    if file is not None:
        raw = file.getvalue()

        if len(raw) > 10 * 1024 * 1024:
            st.error("Maximum file size is 10 MB.")
        else:
            try:
                image = Image.open(file)
                image.load()
                image = image.convert("L")

                new_hash = hashlib.sha256(raw).hexdigest()

                if new_hash != st.session_state.image_hash:
                    st.session_state.results = None

                st.session_state.xray_image = image
                st.session_state.image_hash = new_hash

                st.image(image, width=400)
                st.success("Image ready for AI analysis.")
            except (OSError, ValueError):
                st.error("Invalid image.")

elif page == "AI Analysis":
    st.header("AI Chest X-ray Analysis")

    if st.session_state.xray_image is None:
        st.info("Upload an X-ray image first.")
    else:
        st.image(
            st.session_state.xray_image,
            width=350
        )

        if st.button("Analyze X-ray with AI"):
            try:
                with st.spinner("Running AI inference..."):
                    st.session_state.results = predict(
                        st.session_state.xray_image
                    )
                st.success("AI inference completed.")
            except Exception as exc:
                st.error("AI model could not run.")
                st.exception(exc)

        if st.session_state.results is not None:
            st.subheader("Model output scores")
            st.caption(
                "Scores are not calibrated disease "
                "probabilities or diagnoses."
            )
            for label, score in st.session_state.results:
                st.write(f"{label}: {score:.4f}")

            st.info(
                "A qualified clinician must independently "
                "review these research outputs."
            )

elif page == "Doctor Review":
    st.header("Doctor Review")
    st.info(
        "Doctor authentication and review records "
        "will be implemented next."
    )

elif page == "Model Control":
    st.header("Model Control")
    st.write("Model:", MODEL_ID)
    st.write("Execution: Cloud CPU")
    st.info(
        "Model checksum and approval registry "
        "will be implemented next."
    )

elif page == "Audit Log":
    st.header("Audit Log")
    st.info(
        "Persistent audit logging will be "
        "implemented next."
    )
