
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

def init_state():
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
                value.copy() if isinstance(value, list)
                else value
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
# 3. AUDIT LOG
# ==========================================

def audit(action, details=None):
    event = {
        "event_id": new_id(),
        "timestamp": now(),
        "actor": st.session_state.user or "anonymous",
        "role": st.session_state.role or "anonymous",
        "action": action,
        "details": details or {},
    }

    st.session_state.audit_logs.append(event)


# ==========================================
# 4. AUTHENTICATION
# ==========================================

def verify_login(username, password):
    if username not in ("doctor", "admin"):
        return False

    try:
        data = st.secrets[username]

        if not hmac.compare_digest(
            str(data["username"]), username
        ):
            return False

        salt = bytes.fromhex(str(data["salt"]))
        expected = bytes.fromhex(
            str(data["password_hash"])
        )
        iterations = int(data["iterations"])

        if (
            iterations < 100_000
            or not salt
            or len(expected) != 32
        ):
            return False

        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            iterations
        )

        return hmac.compare_digest(actual, expected)

    except (
        KeyError,
        TypeError,
        ValueError,
        FileNotFoundError
    ):
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

    with Image.open(io.BytesIO(raw)) as source:
        if source.format not in ("JPEG", "PNG"):
            raise ValueError(
                "Only JPEG and PNG are supported."
            )

        if (
            source.width > 5000
            or source.height > 5000
        ):
            raise ValueError(
                "Image dimensions exceed 5000 pixels."
            )

        image = source.convert("L")
        image.load()

    return image


# ==========================================
# 6. AI MODEL
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


def predict(image):
    import torchxrayvision as xrv

    pixels = np.asarray(
        image.convert("L"),
        dtype=np.float32
    )

    pixels = xrv.datasets.normalize(
        pixels, 255
    )[None, :, :]

    pixels = xrv.datasets.XRayCenterCrop()(pixels)
    pixels = xrv.datasets.XRayResizer(224)(pixels)

    tensor = torch.from_numpy(
        np.ascontiguousarray(pixels)
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
            model.pathologies, output
        )
        if np.isfinite(score)
    ]

    return sorted(
        scores,
        key=lambda x: x["score"],
        reverse=True
    )


# ==========================================
# 7. INITIALIZE
# ==========================================

init_state()

st.title(
    "🏥 Sovereign AI Medical Imaging System"
)

st.caption(
    "基於主權 AI 之胸部 X 光影像輔助分析系統"
)

st.warning(
    "Educational demonstration only. "
    "Not for clinical diagnosis. "
    "Use only public, de-identified test images."
)


# ==========================================
# 8. LOGIN
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
            "Missing account settings. "
            "Check Streamlit Cloud → "
            "App settings → Secrets."
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
                {"attempted_username": username}
            )

            st.error(
                "Incorrect username or password."
            )

            if st.session_state.failed_logins >= 5:
                st.warning(
                    "Repeated failures. This demo "
                    "does not implement production-grade "
                    "rate limiting."
                )

    st.stop()


# ==========================================
# 9. ROLE-BASED NAVIGATION
# ==========================================

role = st.session_state.role

st.sidebar.subheader("Control Panel")

st.sidebar.write(
    f"Logged in: **{st.session_state.user}** "
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

    a.metric("Role", role.title())
    b.metric("AI Model", "DenseNet121")

    c.metric(
        "Session Audit Events",
        len(st.session_state.audit_logs)
    )

    st.subheader("Sovereign AI Status")

    st.write(
        "✅ AI inference in the deployed app (Doctor)"
    )

    st.write(
        "✅ Role-gated pages and password-hash "
        "sign-in (demo)"
    )

    st.write(
        "✅ Human review with session-only Audit Log"
    )

    st.write(
        "🟡 Persistent audit database: "
        "not implemented"
    )

    st.write(
        "🟡 Model integrity checking: "
        "not implemented"
    )

    st.write(
        "🟡 DICOM / burned-in patient information "
        "removal: not implemented"
    )

    st.info(
        "Cloud hardware is operated by a third "
        "party; full infrastructure sovereignty "
        "is not claimed."
    )


# ==========================================
# 11. UPLOAD X-RAY
# ==========================================

elif page == "Upload X-ray" and role == "doctor":
    st.header("Upload X-ray")

    st.info(
        "Upload public, already de-identified "
        "X-ray images only."
    )

    uploaded = st.file_uploader(
        "PNG/JPEG image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded is not None:
        try:
            raw = uploaded.getvalue()
            image = read_image(raw)

            new_hash = hashlib.sha256(
                raw
            ).hexdigest()

            if (
                new_hash != st.session_state.image_hash
                or not st.session_state.image_id
                or st.session_state.xray_image is None
            ):
                clear_case()

                st.session_state.xray_image = image
                st.session_state.image_hash = new_hash
                st.session_state.image_id = new_id()

                audit(
                    "IMAGE_UPLOADED",
                    {
                        "image_id":
                            st.session_state.image_id,
                        "image_sha256": new_hash,
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
            "Current Image ID:",
            st.session_state.image_id
        )


# ==========================================
# 12. AI ANALYSIS
# ==========================================

elif page == "AI Analysis" and role == "doctor":
    st.header("AI Chest X-ray Analysis")

    if st.session_state.xray_image is None:
        st.info(
            "Upload an X-ray image first."
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
                    scores = predict(
                        st.session_state.xray_image
                    )

                if not st.session_state.image_id:
                    raise ValueError(
                        "Image ID missing. "
                        "Upload the image again."
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
                    "AI inference failed."
                )
                st.exception(exc)

        if st.session_state.results is not None:
            st.subheader(
                "AI Model Output Scores"
            )

            st.caption(
                "These scores are not calibrated "
                "disease probabilities or diagnoses."
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
        "Demo role only. No professional medical "
        "credentials are verified."
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

        with st.form("doctor_review_form"):
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
                    "Please enter review notes."
                )

            elif (
                not st.session_state.image_id
                or not st.session_state.analysis_id
            ):
                st.error(
                    "Missing image/analysis ID; "
                    "re-upload and analyze again."
                )

            else:
                record = {
                    "review_id": new_id(),
                    "image_id":
                        st.session_state.image_id,
                    "analysis_id":
                        st.session_state.analysis_id,
                    "reviewer": st.session_state.user,
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
                    "Review successfully recorded "
                    "for this session."
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
                data=json.dumps(
                    st.session_state.review,
                    ensure_ascii=False,
                    indent=2
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
        "Model:",
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
            "Admin role: model information access."
        )

        st.warning(
            "Model activation, approval, and weight "
            "SHA-256 verification are not implemented."
        )

    else:
        st.info(
            "Doctor role: read-only model information."
        )


# ==========================================
# 15. AUDIT LOG - FIX KEYERROR
# ==========================================

elif page == "Audit Log" and role == "admin":
    st.header(
        "Audit Log – 稽核紀錄"
    )

    st.warning(
        "Demo only. Logs belong to this Streamlit "
        "session, are not shared across users, "
        "and are not persisted."
    )

    logs = st.session_state.audit_logs

    st.metric(
        "Events in this session",
        len(logs)
    )

    if logs:
        # FIX: Read legacy events safely.
        # Older records may lack actor or role.

        rows = []

        for item in reversed(logs):
            rows.append({
                "timestamp": item.get(
                    "timestamp", ""
                ),
                "actor": item.get(
                    "actor", "legacy/unknown"
                ),
                "role": item.get(
                    "role", "unknown"
                ),
                "action": item.get(
                    "action", "UNKNOWN"
                ),
                "event_id": item.get(
                    "event_id", ""
                )
            })

        st.dataframe(
            rows,
            hide_index=True,
            use_container_width=True
        )

        st.subheader(
            "Event Details"
        )

        for item in reversed(logs):
            label = (
                f"{item.get('timestamp', '')} | "
                f"{item.get('action', 'UNKNOWN')}"
            )

            with st.expander(label):
                st.json(item)

        st.download_button(
            "Download Audit Log (JSON)",
            data=json.dumps(
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
            "Doctor events in other sessions "
            "are not visible here."
        )
