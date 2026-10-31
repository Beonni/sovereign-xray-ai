
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
# 1. SYSTEM CONFIGURATION
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

defaults = {
    "xray_image": None,
    "image_hash": None,
    "image_id": None,
    "results": None,
    "analysis_id": None,
    "review": None,
    "audit_logs": [],
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = (
            [] if isinstance(value, list) else value
        )

# ==========================================
# 3. AUDIT LOG FUNCTIONS
# ==========================================

def utc_now():
    return datetime.now(timezone.utc).isoformat()


def add_audit_log(action, details):
    event = {
        "event_id": str(uuid.uuid4()),
        "timestamp": utc_now(),
        "action": action,
        "details": details
    }

    st.session_state.audit_logs.append(event)
    return event


# ==========================================
# 4. AI MODEL
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

    # Convert image to grayscale
    img = np.asarray(
        image.convert("L"),
        dtype=np.float32
    )

    # Normalize to TorchXRayVision range
    img = xrv.datasets.normalize(img, 255)

    # Add channel dimension
    img = img[None, :, :]

    # Preprocessing
    img = xrv.datasets.XRayCenterCrop()(img)
    img = xrv.datasets.XRayResizer(224)(img)

    tensor = torch.from_numpy(
        np.ascontiguousarray(img)
    ).unsqueeze(0).float()

    model = load_model()

    with torch.inference_mode():
        outputs = model(tensor)[0].cpu().numpy()

    results = [
        {
            "finding": name,
            "score": float(score)
        }
        for name, score in zip(
            model.pathologies,
            outputs
        )
        if np.isfinite(score)
    ]

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return results


# ==========================================
# 5. IMAGE VALIDATION
# ==========================================

def validate_image(raw):
    if len(raw) > MAX_FILE_SIZE:
        raise ValueError(
            "Image exceeds maximum size of 10 MB."
        )

    if not raw:
        raise ValueError("Empty image file.")

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
                "Image dimensions are too large."
            )

        # Decode pixels and discard source metadata
        image = source.convert("L")
        image.load()

    return image


# ==========================================
# 6. MAIN UI
# ==========================================

st.title("🏥 Sovereign AI Medical Imaging System")

st.caption(
    "基於主權 AI 之胸部 X 光影像輔助分析系統"
)

st.warning(
    "Educational research demonstration only. "
    "AI outputs are not validated for clinical diagnosis."
)

st.sidebar.title("Sovereign AI Control Panel")

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
st.sidebar.caption("Environment: Streamlit Cloud CPU")
st.sidebar.caption("Model: " + MODEL_ID)


# ==========================================
# 7. DASHBOARD
# ==========================================

if page == "Dashboard":
    st.header("System Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "AI Environment",
        "Cloud CPU"
    )

    col2.metric(
        "AI Model",
        "DenseNet121"
    )

    col3.metric(
        "Audit Events",
        len(st.session_state.audit_logs)
    )

    st.subheader("Sovereign AI Controls")

    st.write(
        "✅ AI inference runs inside the deployed application"
    )

    st.write(
        "✅ Model identifier is explicitly configured"
    )

    st.write(
        "✅ Human review workflow (demo)"
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
        "The cloud infrastructure is operated by "
        "a third-party provider. Full infrastructure "
        "sovereignty is not claimed."
    )


# ==========================================
# 8. UPLOAD X-RAY
# ==========================================

elif page == "Upload X-ray":
    st.header("Upload Chest X-ray")

    st.info(
        "Only upload public, de-identified "
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

            image_hash = hashlib.sha256(
                raw
            ).hexdigest()

            # Reset analysis when a NEW image is uploaded
            if image_hash != st.session_state.image_hash:
                st.session_state.xray_image = image
                st.session_state.image_hash = image_hash
                st.session_state.image_id = str(
                    uuid.uuid4()
                )

                st.session_state.results = None
                st.session_state.analysis_id = None
                st.session_state.review = None

                add_audit_log(
                    "IMAGE_UPLOADED",
                    {
                        "image_id":
                            st.session_state.image_id,
                        "image_sha256": image_hash,
                        "size_bytes": len(raw)
                    }
                )

            st.image(
                st.session_state.xray_image,
                caption="X-ray Preview",
                width=400
            )

            st.success(
                "Image ready for AI analysis."
            )

            st.caption(
                "Image ID: "
                + st.session_state.image_id
            )

        except (
            UnidentifiedImageError,
            OSError,
            ValueError
        ) as exc:
            st.error(f"Upload failed: {exc}")


# ==========================================
# 9. AI ANALYSIS
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

                analysis_id = str(uuid.uuid4())

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
                    "AI inference completed successfully!"
                )

            except Exception as exc:
                st.error(
                    "AI model could not run."
                )
                st.exception(exc)

        if st.session_state.results is not None:
            st.subheader("AI Model Output Scores")

            st.caption(
                "These are model output scores, "
                "not calibrated disease probabilities."
            )

            st.write(
                "Analysis ID:",
                st.session_state.analysis_id
            )

            st.dataframe(
                st.session_state.results,
                use_container_width=True,
                hide_index=True
            )

            st.info(
                "A qualified clinician must independently "
                "review the model outputs."
            )


# ==========================================
# 10. DOCTOR REVIEW
# ==========================================

elif page == "Doctor Review":
    st.header("Doctor Review – 醫師審查")

    st.warning(
        "Demonstration workflow only. "
        "Reviewer identity is not authenticated."
    )

    if st.session_state.results is None:
        st.info(
            "Please complete AI Analysis first."
        )

    else:
        st.subheader("Patient Image – Demo")

        st.image(
            st.session_state.xray_image,
            width=350
        )

        st.subheader("AI Results")

        st.write(
            "Analysis ID:",
            st.session_state.analysis_id
        )

        st.dataframe(
            st.session_state.results,
            use_container_width=True,
            hide_index=True
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

            else:
                record = {
                    "review_id": str(uuid.uuid4()),
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
            st.subheader("Latest Review")

            st.json(
                st.session_state.review
            )


# ==========================================
# 11. MODEL CONTROL
# ==========================================

elif page == "Model Control":
    st.header("Model Management")

    st.write(
        "Model Name: DenseNet121"
    )

    st.write(
        "Model ID:",
        MODEL_ID
    )

    st.write(
        "Execution Environment: Cloud CPU"
    )

    st.write(
        "Framework: PyTorch / TorchXRayVision"
    )

    st.warning(
        "Model checksum verification, "
        "version approval and authorized model "
        "switching are not implemented yet."
    )


# ==========================================
# 12. AUDIT LOG
# ==========================================

elif page == "Audit Log":
    st.header("Audit Log – 稽核紀錄")

    st.warning(
        "Session-only audit history. "
        "Logs may be lost when the session ends. "
        "This is not a tamper-proof audit system."
    )

    logs = st.session_state.audit_logs

    st.metric(
        "Total Audit Events",
        len(logs)
    )

    if logs:
        summary = [
            {
                "timestamp": event["timestamp"],
                "action": event["action"],
                "event_id": event["event_id"]
            }
            for event in reversed(logs)
        ]

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Event Details")

        for event in reversed(logs):
            with st.expander(
                f"{event['timestamp']} | "
                f"{event['action']}"
            ):
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
            "No audit events recorded in this session."
        )
