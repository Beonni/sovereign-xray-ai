
import streamlit as st
from PIL import Image, UnidentifiedImageError
import numpy as np
import hashlib
import torch
import json
import uuid
import io
from datetime import datetime, timezone

# ==========================================
# 1. CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Sovereign X-ray AI",
    page_icon="🏥",
    layout="wide"
)

MODEL_ID = "densenet121-res224-all"
MAX_FILE_SIZE = 10 * 1024 * 1024

# ==========================================
# 2. SESSION STATE
# ==========================================

DEFAULTS = {
    "xray_image": None,
    "image_hash": None,
    "image_id": None,
    "results": None,
    "analysis_id": None,
    "review": None,
    "audit_logs": [],
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = (
            [] if isinstance(value, list) else value
        )


def utc_now():
    return datetime.now(
        timezone.utc
    ).isoformat()


def new_id():
    return str(uuid.uuid4())


# ==========================================
# 3. AUDIT LOG
# ==========================================

def add_audit_log(action, details):
    event = {
        "event_id": new_id(),
        "timestamp": utc_now(),
        "action": action,
        "details": details
    }

    st.session_state.audit_logs.append(event)
    return event


# ==========================================
# 4. IMAGE VALIDATION
# ==========================================

def validate_image(raw):
    if not raw:
        raise ValueError("Image file is empty.")

    if len(raw) > MAX_FILE_SIZE:
        raise ValueError(
            "Maximum image size is 10 MB."
        )

    with Image.open(io.BytesIO(raw)) as source:
        if source.format not in ("PNG", "JPEG"):
            raise ValueError(
                "Only PNG and JPEG images are supported."
            )

        if (
            source.width > 5000
            or source.height > 5000
        ):
            raise ValueError(
                "Image dimensions exceed the limit."
            )

        # Decode pixels into a new grayscale image.
        # Original metadata is not stored in session.
        image = source.convert("L")
        image.load()

    return image


# ==========================================
# 5. AI MODEL
# ==========================================

@st.cache_resource(
    show_spinner="Loading X-ray AI model..."
)
def load_model():
    import torchxrayvision as xrv

    model = xrv.models.DenseNet(
        weights=MODEL_ID
    )

    model.eval()
    return model


def predict_xray(image):
    import torchxrayvision as xrv

    img = np.asarray(
        image.convert("L"),
        dtype=np.float32
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

    results = []

    for name, score in zip(
        model.pathologies,
        outputs
    ):
        if np.isfinite(score):
            results.append({
                "finding": name,
                "score": float(score)
            })

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return results


# ==========================================
# 6. IMAGE MANAGEMENT
# ==========================================

def register_image(image, raw):
    image_hash = hashlib.sha256(
        raw
    ).hexdigest()

    previous_hash = st.session_state.image_hash
    previous_id = st.session_state.image_id

    # FIX: Generate ID when image changes
    # OR when an older session has a missing ID.
    is_new = (
        image_hash != previous_hash
        or previous_id is None
        or st.session_state.xray_image is None
    )

    if is_new:
        image_id = new_id()

        st.session_state.xray_image = image
        st.session_state.image_hash = image_hash
        st.session_state.image_id = image_id

        # Clear previous analysis and review
        st.session_state.results = None
        st.session_state.analysis_id = None
        st.session_state.review = None

        add_audit_log(
            "IMAGE_UPLOADED",
            {
                "image_id": image_id,
                "image_sha256": image_hash,
                "size_bytes": len(raw),
                "model": MODEL_ID
            }
        )

    return st.session_state.image_id


# ==========================================
# 7. MAIN INTERFACE
# ==========================================

st.title(
    "🏥 Sovereign AI Medical Imaging System"
)

st.caption(
    "基於主權 AI 之胸部 X 光影像輔助分析系統"
)

st.warning(
    "Educational demonstration only. "
    "Not intended for clinical diagnosis."
)

st.sidebar.title(
    "Sovereign AI Control Panel"
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

st.sidebar.divider()
st.sidebar.caption("Environment: Cloud CPU")
st.sidebar.caption("Model: " + MODEL_ID)


# ==========================================
# 8. DASHBOARD
# ==========================================

if page == "Dashboard":
    st.header("System Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("AI Environment", "Cloud CPU")
    col2.metric("AI Model", "DenseNet121")
    col3.metric(
        "Audit Events",
        len(st.session_state.audit_logs)
    )

    st.subheader("Sovereign AI Status")

    st.success(
        "AI inference is available inside "
        "the deployed application."
    )

    st.write(
        "✅ X-ray image upload and validation"
    )

    st.write(
        "✅ AI image analysis"
    )

    st.write(
        "✅ Human decision workflow (demo)"
    )

    st.write(
        "✅ Session-based audit logging"
    )

    st.write(
        "🟡 Authentication and RBAC: pending"
    )

    st.write(
        "🟡 Persistent audit storage: pending"
    )

    st.write(
        "🟡 Model integrity verification: pending"
    )

    st.info(
        "Cloud infrastructure is managed by "
        "a third-party provider. "
        "Full infrastructure sovereignty "
        "is not claimed."
    )


# ==========================================
# 9. UPLOAD X-RAY
# ==========================================

elif page == "Upload X-ray":
    st.header("Upload Chest X-ray")

    st.info(
        "Use only public, de-identified "
        "educational test images."
    )

    file = st.file_uploader(
        "Choose X-ray image",
        type=["png", "jpg", "jpeg"]
    )

    if file is not None:
        try:
            raw = file.getvalue()
            image = validate_image(raw)

            image_id = register_image(
                image,
                raw
            )

            st.image(
                st.session_state.xray_image,
                caption="Uploaded X-ray",
                width=400
            )

            st.success(
                "Image ready for AI analysis!"
            )

            st.write("Image ID:", image_id)

            st.caption(
                "SHA-256: "
                + st.session_state.image_hash
            )

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
            Image.DecompressionBombError
        ) as exc:
            st.error(
                f"Upload failed: {exc}"
            )

    elif st.session_state.xray_image is not None:
        st.subheader("Current Session Image")

        st.image(
            st.session_state.xray_image,
            width=350
        )

        st.write(
            "Image ID:",
            st.session_state.image_id
        )


# ==========================================
# 10. AI ANALYSIS
# ==========================================

elif page == "AI Analysis":
    st.header("AI Chest X-ray Analysis")

    if st.session_state.xray_image is None:
        st.info(
            "Please upload an X-ray image first."
        )

    else:
        st.image(
            st.session_state.xray_image,
            width=350
        )

        st.write(
            "Image ID:",
            st.session_state.image_id
        )

        st.write(
            "Model:",
            MODEL_ID
        )

        if st.button(
            "Analyze X-ray with AI",
            type="primary"
        ):
            try:
                with st.spinner(
                    "Running AI inference..."
                ):
                    prediction = predict_xray(
                        st.session_state.xray_image
                    )

                # Ensure image ID exists
                if st.session_state.image_id is None:
                    raise ValueError(
                        "Image ID is missing. "
                        "Please upload the image again."
                    )

                analysis_id = new_id()

                st.session_state.results = prediction
                st.session_state.analysis_id = analysis_id
                st.session_state.review = None

                add_audit_log(
                    "AI_INFERENCE_COMPLETED",
                    {
                        "analysis_id": analysis_id,
                        "image_id":
                            st.session_state.image_id,
                        "image_sha256":
                            st.session_state.image_hash,
                        "model": MODEL_ID,
                        "scores": prediction
                    }
                )

                st.success(
                    "AI inference completed!"
                )

            except Exception as exc:
                st.error(
                    "AI inference failed."
                )
                st.exception(exc)

        if st.session_state.results is not None:
            st.divider()

            st.subheader("AI Model Output Scores")

            st.caption(
                "Scores are not calibrated disease "
                "probabilities or clinical diagnoses."
            )

            st.write(
                "Analysis ID:",
                st.session_state.analysis_id
            )

            st.dataframe(
                st.session_state.results,
                hide_index=True,
                use_container_width=True
            )

            st.info(
                "AI outputs require independent "
                "review by a qualified clinician."
            )


# ==========================================
# 11. DOCTOR REVIEW
# ==========================================

elif page == "Doctor Review":
    st.header("Doctor Review – 醫師審查")

    st.warning(
        "Demonstration only. Reviewer identity "
        "is not authenticated."
    )

    if st.session_state.results is None:
        st.info(
            "Please complete AI Analysis first."
        )

    else:
        st.subheader("X-ray Image")

        st.image(
            st.session_state.xray_image,
            width=350
        )

        st.write(
            "Image ID:",
            st.session_state.image_id
        )

        st.write(
            "Analysis ID:",
            st.session_state.analysis_id
        )

        st.subheader("AI Results")

        st.dataframe(
            st.session_state.results,
            hide_index=True,
            use_container_width=True
        )

        st.divider()

        st.subheader("Human Decision Control")

        with st.form("doctor_review_form"):
            reviewer = st.text_input(
                "Reviewer name (demo only)"
            )

            decision = st.radio(
                "Review decision",
                [
                    "Accept AI findings",
                    "Reject AI findings",
                    "Modify AI findings"
                ]
            )

            notes = st.text_area(
                "Review notes / 修改原因"
            )

            submitted = st.form_submit_button(
                "Confirm Review",
                type="primary"
            )

        if submitted:
            if not reviewer.strip():
                st.error(
                    "Please enter reviewer name."
                )

            elif not notes.strip():
                st.error(
                    "Please enter review notes."
                )

            elif st.session_state.image_id is None:
                st.error(
                    "Image ID is missing. "
                    "Please upload the image again."
                )

            else:
                record = {
                    "review_id": new_id(),
                    "analysis_id":
                        st.session_state.analysis_id,
                    "image_id":
                        st.session_state.image_id,
                    "reviewer": reviewer.strip(),
                    "decision": decision,
                    "notes": notes.strip(),
                    "model": MODEL_ID,
                    "timestamp": utc_now()
                }

                st.session_state.review = record

                add_audit_log(
                    "DOCTOR_REVIEW_COMPLETED",
                    record
                )

                st.success(
                    "Review successfully recorded!"
                )

        if st.session_state.review is not None:
            st.divider()

            st.subheader("Latest Review")

            st.json(
                st.session_state.review
            )

            st.download_button(
                "Download Review JSON",
                data=json.dumps(
                    st.session_state.review,
                    ensure_ascii=False,
                    indent=2
                ),
                file_name="doctor_review.json",
                mime="application/json"
            )


# ==========================================
# 12. MODEL CONTROL
# ==========================================

elif page == "Model Control":
    st.header("Model Management")

    st.write("Model Name: DenseNet121")
    st.write("Model ID:", MODEL_ID)

    st.write(
        "Framework: PyTorch / TorchXRayVision"
    )

    st.write(
        "Execution Environment: Streamlit Cloud CPU"
    )

    st.write(
        "Weights source: TorchXRayVision pretrained model"
    )

    st.warning(
        "Model weight checksum, approval registry "
        "and authorized model switching "
        "are not implemented yet."
    )


# ==========================================
# 13. AUDIT LOG
# ==========================================

elif page == "Audit Log":
    st.header("Audit Log – 稽核紀錄")

    st.warning(
        "Session-only audit records. "
        "Events may disappear when the session ends. "
        "This is not tamper-proof storage."
    )

    logs = st.session_state.audit_logs

    st.metric(
        "Total Audit Events",
        len(logs)
    )

    if logs:
        summary = []

        for event in reversed(logs):
            details = event.get("details", {})

            summary.append({
                "timestamp": event["timestamp"],
                "action": event["action"],
                "image_id": details.get(
                    "image_id", ""
                ),
                "analysis_id": details.get(
                    "analysis_id", ""
                ),
                "event_id": event["event_id"]
            })

        st.dataframe(
            summary,
            hide_index=True,
            use_container_width=True
        )

        st.subheader("Event Details")

        for event in reversed(logs):
            title = (
                event["timestamp"]
                + " | "
                + event["action"]
            )

            with st.expander(title):
                st.json(event)

        json_data = json.dumps(
            logs,
            ensure_ascii=False,
            indent=2
        )

        st.download_button(
            "Download Audit Log (JSON)",
            data=json_data,
            file_name="sovereign_ai_audit.json",
            mime="application/json"
        )

    else:
        st.info(
            "No audit events recorded "
            "in this session."
        )
