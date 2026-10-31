
import hashlib
import hmac
import io
import json
import uuid
from datetime import datetime, timezone

import numpy as np
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

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


def now():
    return datetime.now(timezone.utc).isoformat()


def new_id():
    return str(uuid.uuid4())


# ==========================================
# 2. SESSION MANAGEMENT
# ==========================================

def initialize():
    defaults = {
        "user": None,
        "role": None,
        "xray_image": None,
        "image_hash": None,
        "image_id": None,
        "results": None,
        "analysis_id": None,
        "review": None,
        "audit_logs": [],
        "failed_logins": 0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = (
                [] if isinstance(value, list) else value
            )


def clear_case():
    for key in (
        "xray_image",
        "image_hash",
        "image_id",
        "results",
        "analysis_id",
        "review",
    ):
        st.session_state[key] = None


# ==========================================
# 3. AUDIT LOG FUNCTIONS
# ==========================================

def audit(action, details=None):
    st.session_state.audit_logs.append({
        "event_id": new_id(),
        "timestamp": now(),
        "actor": st.session_state.user or "anonymous",
        "role": st.session_state.role or "anonymous",
        "action": action,
        "details": details or {},
    })


# ==========================================
# 4. AUTHENTICATION
# ==========================================

def verify_login(username, password):
    if username not in ("doctor", "admin"):
        return False

    try:
        entry = st.secrets[username]

        if not hmac.compare_digest(
            str(entry["username"]), username
        ):
            return False

        salt = bytes.fromhex(entry["salt"])
        expected = bytes.fromhex(entry["password_hash"])
        iterations = int(entry["iterations"])

        if iterations < 100000:
            return False

        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            iterations
        )

        return hmac.compare_digest(actual, expected)

    except (KeyError, ValueError, TypeError):
        return False


def logout():
    audit("LOGOUT")
    clear_case()

    st.session_state.user = None
    st.session_state.role = None

    st.rerun()


# ==========================================
# 5. IMAGE VALIDATION
# ==========================================

def read_image(raw):
    if not raw or len(raw) > MAX_FILE_SIZE:
        raise ValueError(
            "Image must be nonempty and at most 10 MB."
        )

    with Image.open(io.BytesIO(raw)) as original:
        if original.format not in ("PNG", "JPEG"):
            raise ValueError(
                "Only PNG/JPEG images are supported."
            )

        if (
            original.width > 5000
            or original.height > 5000
        ):
            raise ValueError(
                "Image dimensions exceed 5000 pixels."
            )

        # Rebuild image from pixels.
        # Original EXIF metadata is not retained.
        image = original.convert("L")
        image.load()

    return image


# ==========================================
# 6. AI MODEL
# ==========================================

@st.cache_resource(
    show_spinner="Loading pretrained X-ray model..."
)
def load_model():
    import torchxrayvision as xrv

    model = xrv.models.DenseNet(
        weights=MODEL_ID
    )

    model.eval()

    return model


def predict(image):
    import torchxrayvision as xrv

    data = np.asarray(
        image.convert("L"),
        dtype=np.float32
    )

    data = xrv.datasets.normalize(data, 255)
    data = data[None, :, :]

    data = xrv.datasets.XRayCenterCrop()(data)
    data = xrv.datasets.XRayResizer(224)(data)

    tensor = torch.from_numpy(
        np.ascontiguousarray(data)
    ).unsqueeze(0).float()

    model = load_model()

    with torch.inference_mode():
        output = model(tensor)[0].cpu().numpy()

    scores = [
        {
            "finding": label,
            "score": float(score)
        }
        for label, score in zip(
            model.pathologies,
            output
        )
        if np.isfinite(score)
    ]

    return sorted(
        scores,
        key=lambda item: item["score"],
        reverse=True
    )


# ==========================================
# 7. INITIALIZE APPLICATION
# ==========================================

initialize()

st.title(
    "🏥 Sovereign AI Medical Imaging System"
)

st.caption(
    "基於主權 AI 之胸部 X 光影像輔助分析系統"
)

st.warning(
    "For educational demonstration only. "
    "Not for clinical use. "
    "Use only public, de-identified test images."
)


# ==========================================
# 8. LOGIN PAGE
# ==========================================

if st.session_state.user is None:

    st.subheader("Sign in – 身分驗證")

    try:
        configured = all(
            all(
                key in st.secrets[role]
                for key in (
                    "username",
                    "salt",
                    "password_hash",
                    "iterations"
                )
            )
            for role in ("doctor", "admin")
        )

    except (KeyError, FileNotFoundError):
        configured = False

    if not configured:
        st.error(
            "Account configuration is missing. "
            "Check Streamlit App settings → Secrets."
        )
        st.stop()

    with st.form("login_form"):
        username = st.text_input(
            "Username"
        ).strip().lower()

        password = st.text_input(
            "Password",
            type="password"
        )

        submitted = st.form_submit_button(
            "Login",
            type="primary"
        )

    if submitted:

        if verify_login(username, password):

            st.session_state.user = username
            st.session_state.role = username
            st.session_state.failed_logins = 0

            clear_case()

            audit("LOGIN_SUCCESS")

            st.rerun()

        else:
            st.session_state.failed_logins += 1

            audit(
                "LOGIN_FAILED",
                {
                    "attempted_username": username
                }
            )

            st.error(
                "Incorrect username or password."
            )

            if st.session_state.failed_logins >= 5:
                st.warning(
                    "Multiple failed attempts. "
                    "This demo does not provide "
                    "production-grade rate limiting."
                )

    st.stop()


# ==========================================
# 9. ROLE-BASED NAVIGATION
# ==========================================

role = st.session_state.role

st.sidebar.subheader("Control Panel")

st.sidebar.write(
    f"Signed in as **{st.session_state.user}** "
    f"({role})"
)

st.sidebar.button(
    "Log out",
    on_click=logout
)

if role == "doctor":
    pages = [
        "Dashboard",
        "Upload X-ray",
        "AI Analysis",
        "Doctor Review",
        "Model Control"
    ]
else:
    pages = [
        "Dashboard",
        "Model Control",
        "Audit Log"
    ]

page = st.sidebar.radio(
    "Navigation",
    pages
)

st.sidebar.caption(
    f"Model: {MODEL_ID}"
)


# ==========================================
# 10. DASHBOARD
# ==========================================

if page == "Dashboard":

    st.header("System Dashboard")

    a, b, c = st.columns(3)

    a.metric(
        "Role",
        role.title()
    )

    b.metric(
        "Model",
        "DenseNet121"
    )

    c.metric(
        "Session audit events",
        len(st.session_state.audit_logs)
    )

    st.subheader("Sovereign AI Controls")

    st.write(
        "✅ Real AI inference (Doctor only)"
    )

    st.write(
        "✅ Password-hash sign-in and "
        "role-gated pages (demo)"
    )

    st.write(
        "✅ Doctor review and session audit trail"
    )

    st.write(
        "🟡 Persistent multi-user audit logging: "
        "not yet implemented"
    )

    st.write(
        "🟡 Verified model weights and controlled "
        "deployment: not yet implemented"
    )

    st.write(
        "🟡 DICOM/pixel de-identification: "
        "not implemented"
    )

    st.info(
        "Cloud infrastructure is managed by "
        "the hosting provider. "
        "This is not full infrastructure sovereignty."
    )


# ==========================================
# 11. UPLOAD X-RAY
# ==========================================

elif page == "Upload X-ray" and role == "doctor":

    st.header("Upload X-ray")

    st.info(
        "Only upload public, already de-identified "
        "teaching images. No patient records."
    )

    uploaded = st.file_uploader(
        "PNG or JPEG",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded is not None:

        try:
            raw = uploaded.getvalue()
            image = read_image(raw)

            image_hash = hashlib.sha256(
                raw
            ).hexdigest()

            if (
                image_hash != st.session_state.image_hash
                or not st.session_state.image_id
                or st.session_state.xray_image is None
            ):

                clear_case()

                st.session_state.xray_image = image
                st.session_state.image_hash = image_hash
                st.session_state.image_id = new_id()

                audit(
                    "IMAGE_UPLOADED",
                    {
                        "image_id":
                            st.session_state.image_id,
                        "sha256": image_hash,
                        "size_bytes": len(raw)
                    }
                )

            st.image(
                st.session_state.xray_image,
                width=380
            )

            st.success(
                "Image ready for analysis."
            )

            st.write(
                "Image ID:",
                st.session_state.image_id
            )

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
            Image.DecompressionBombError
        ) as exc:

            st.error(
                f"Invalid image: {exc}"
            )

    elif st.session_state.xray_image is not None:

        st.image(
            st.session_state.xray_image,
            width=340
        )

        st.write(
            "Current image ID:",
            st.session_state.image_id
        )


# ==========================================
# 12. AI ANALYSIS
# ==========================================

elif page == "AI Analysis" and role == "doctor":

    st.header("AI Analysis")

    if st.session_state.xray_image is None:
        st.info(
            "Upload an X-ray first."
        )

    else:

        st.image(
            st.session_state.xray_image,
            width=340
        )

        st.write(
            "Image ID:",
            st.session_state.image_id
        )

        if st.button(
            "Analyze X-ray with AI",
            type="primary"
        ):

            try:
                scores = predict(
                    st.session_state.xray_image
                )

                analysis_id = new_id()

                st.session_state.results = scores
                st.session_state.analysis_id = analysis_id
                st.session_state.review = None

                audit(
                    "AI_INFERENCE_COMPLETED",
                    {
                        "image_id":
                            st.session_state.image_id,
                        "analysis_id": analysis_id,
                        "model": MODEL_ID,
                        "scores": scores
                    }
                )

                st.success(
                    "AI inference completed."
                )

            except Exception as exc:
                st.error(
                    "Inference failed. "
                    "See the application logs."
                )
                st.exception(exc)

        if st.session_state.results is not None:

            st.subheader(
                "Model Output Scores"
            )

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


# ==========================================
# 13. DOCTOR REVIEW
# ==========================================

elif page == "Doctor Review" and role == "doctor":

    st.header(
        "Doctor Review – 醫師審查"
    )

    st.caption(
        "Authenticated demo role; this account "
        "does not establish medical credentials."
    )

    if st.session_state.results is None:

        st.info(
            "Complete AI Analysis first."
        )

    else:

        st.image(
            st.session_state.xray_image,
            width=340
        )

        st.write(
            "Image ID:",
            st.session_state.image_id
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

        with st.form("review_form"):

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

            if not notes.strip():

                st.error(
                    "Review notes are required."
                )

            else:

                record = {
                    "review_id": new_id(),
                    "image_id":
                        st.session_state.image_id,
                    "analysis_id":
                        st.session_state.analysis_id,
                    "reviewer":
                        st.session_state.user,
                    "decision": decision,
                    "notes": notes.strip(),
                    "model": MODEL_ID,
                    "timestamp": now()
                }

                st.session_state.review = record

                audit(
                    "DOCTOR_REVIEW_COMPLETED",
                    record
                )

                st.success(
                    "Review recorded for this session."
                )

        if st.session_state.review is not None:

            st.subheader(
                "Latest Review"
            )

            st.json(
                st.session_state.review
            )

            st.download_button(
                "Download Review JSON",
                json.dumps(
                    st.session_state.review,
                    indent=2,
                    ensure_ascii=False
                ),
                file_name="doctor_review.json",
                mime="application/json"
            )


# ==========================================
# 14. MODEL CONTROL
# ==========================================

elif page == "Model Control":

    st.header(
        "Model Control – 模型管理"
    )

    st.write(
        "Model ID:",
        MODEL_ID
    )

    st.write(
        "Framework: PyTorch / TorchXRayVision"
    )

    st.write(
        "Runtime: Streamlit Community Cloud CPU"
    )

    if role == "admin":

        st.success(
            "Admin has access to the model "
            "information page."
        )

        st.warning(
            "Model activation, approval and weight "
            "checksum verification are not yet implemented."
        )

    else:

        st.info(
            "Read-only model information. "
            "Admin manages deployment."
        )


# ==========================================
# 15. AUDIT LOG
# ==========================================

elif page == "Audit Log" and role == "admin":

    st.header(
        "Audit Log – 稽核紀錄"
    )

    st.warning(
        "Demo only: these logs are stored in "
        "the current browser session, NOT shared "
        "across users or persisted."
    )

    logs = st.session_state.audit_logs

    st.metric(
        "Events in this session",
        len(logs)
    )

    if logs:

        st.dataframe(
            [
                {
                    "timestamp": item["timestamp"],
                    "actor": item["actor"],
                    "action": item["action"],
                    "event_id": item["event_id"]
                }
                for item in reversed(logs)
            ],
            hide_index=True,
            use_container_width=True
        )

        for event in reversed(logs):

            with st.expander(
                f"{event['timestamp']} | "
                f"{event['action']}"
            ):
                st.json(event)

        st.download_button(
            "Download Audit Log (JSON)",
            json.dumps(
                logs,
                indent=2,
                ensure_ascii=False
            ),
            file_name="sovereign_ai_audit.json",
            mime="application/json"
        )

    else:

        st.info(
            "No events in this session. "
            "Doctor events from a separate login "
            "are not visible here."
        )
