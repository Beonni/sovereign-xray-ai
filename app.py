
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
from supabase import create_client


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
        "last_audit_error": None,
        "failed_logins": 0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = (
                value.copy()
                if isinstance(value, list)
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
# 3. SUPABASE CONNECTION
# ==========================================

@st.cache_resource(show_spinner=False)
def supabase_client(url, key):
    return create_client(url, key)


def get_db():
    url = str(
        st.secrets.get("SUPABASE_URL", "")
    ).strip()

    key = str(
        st.secrets.get("SUPABASE_SECRET_KEY", "")
    ).strip()

    if not url.startswith("https://") or not key:
        raise RuntimeError(
            "Supabase Secrets configuration "
            "is missing or invalid."
        )

    return supabase_client(url, key)


# ==========================================
# 4. PERSISTENT AUDIT LOG
# ==========================================

def audit(action, details=None):
    """
    Save an audit event to Supabase.
    If saving fails, keep a local unsaved
    record and report the failure.
    """

    details = details or {}

    event = {
        "event_id": new_id(),
        "created_at": now(),
        "actor": (
            st.session_state.user or "anonymous"
        ),
        "role": (
            st.session_state.role or "anonymous"
        ),
        "action": action,
        "image_id": details.get("image_id"),
        "analysis_id": details.get("analysis_id"),
        "details": details,
    }

    try:
        get_db().table(
            "audit_logs"
        ).insert(event).execute()

    except Exception:
        st.session_state.last_audit_error = (
            "Could not save event to Supabase. "
            "Check configuration and Cloud Logs."
        )

        st.session_state.audit_logs.append({
            **event,
            "sync_status": "NOT_SAVED"
        })

        return False

    st.session_state.last_audit_error = None

    st.session_state.audit_logs.append({
        **event,
        "sync_status": "SAVED"
    })

    return True


# ==========================================
# 5. AUTHENTICATION
# ==========================================

def verify_login(username, password):
    if username not in ("doctor", "admin"):
        return False

    if not password:
        return False

    try:
        data = st.secrets[username]

        if not hmac.compare_digest(
            str(data["username"]),
            username
        ):
            return False

        salt = bytes.fromhex(
            str(data["salt"])
        )

        expected = bytes.fromhex(
            str(data["password_hash"])
        )

        iterations = int(
            data["iterations"]
        )

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

        return hmac.compare_digest(
            actual,
            expected
        )

    except (
        KeyError,
        ValueError,
        TypeError,
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
# 6. IMAGE VALIDATION
# ==========================================

def read_image(raw):
    if not raw or len(raw) > MAX_FILE_SIZE:
        raise ValueError(
            "File must be nonempty "
            "and at most 10 MB."
        )

    with Image.open(
        io.BytesIO(raw)
    ) as source:

        if source.format not in (
            "PNG",
            "JPEG"
        ):
            raise ValueError(
                "Only PNG/JPEG supported."
            )

        if max(source.size) > 5000:
            raise ValueError(
                "Image dimensions "
                "exceed the limit."
            )

        source.load()

        image = source.convert("L")
        image.load()

    return image


# ==========================================
# 7. AI MODEL
# ==========================================

@st.cache_resource(
    show_spinner="Loading AI X-ray model..."
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

    data = xrv.datasets.normalize(
        data,
        255
    )[None, :, :]

    data = xrv.datasets.XRayCenterCrop()(data)

    data = xrv.datasets.XRayResizer(224)(data)

    tensor = torch.from_numpy(
        np.ascontiguousarray(data)
    ).unsqueeze(0).float()

    model = load_model()

    with torch.inference_mode():
        output = model(tensor)[0].cpu().numpy()

    results = [
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
        results,
        key=lambda item: item["score"],
        reverse=True
    )


# ==========================================
# 8. INITIALIZE APPLICATION
# ==========================================

initialize()

st.title(
    "🏥 Sovereign AI Medical Imaging System"
)

st.caption(
    "基於主權 AI 之胸部 X 光影像輔助分析系統"
)

st.warning(
    "Research and teaching demo only. "
    "Not for clinical diagnosis. "
    "Upload only public, "
    "de-identified sample images."
)


# ==========================================
# 9. LOGIN PAGE
# ==========================================

if st.session_state.user is None:

    st.subheader(
        "Sign in – 身分驗證"
    )

    try:
        configured = all(
            all(
                field in st.secrets[role]
                for field in (
                    "username",
                    "salt",
                    "password_hash",
                    "iterations"
                )
            )
            for role in ("doctor", "admin")
        )

    except (
        KeyError,
        FileNotFoundError
    ):
        configured = False

    if not configured:
        st.error(
            "Missing Doctor/Admin settings "
            "in Streamlit Secrets."
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
                    "attempted_username": username[:50]
                }
            )

            st.error(
                "Incorrect username or password."
            )

            if st.session_state.failed_logins >= 5:
                st.warning(
                    "Demo authentication has no "
                    "robust account lockout "
                    "or rate limiting."
                )

    st.stop()


# ==========================================
# 10. ROLE-BASED NAVIGATION
# ==========================================

role = st.session_state.role

st.sidebar.subheader(
    "Control Panel"
)

st.sidebar.write(
    f"Signed in: **{st.session_state.user}** "
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

elif role == "admin":
    pages = [
        "Dashboard",
        "Model Control",
        "Audit Log"
    ]

else:
    st.error(
        "Unauthorized role"
    )
    st.stop()

page = st.sidebar.radio(
    "Navigation",
    pages
)

st.sidebar.caption(
    f"Model: {MODEL_ID}"
)

if st.session_state.last_audit_error:
    st.error(
        st.session_state.last_audit_error
    )


# ==========================================
# 11. DASHBOARD
# ==========================================

if page == "Dashboard":

    st.header("Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Role",
        role.title()
    )

    col2.metric(
        "AI model",
        "DenseNet121"
    )

    col3.metric(
        "Local session events",
        len(st.session_state.audit_logs)
    )

    st.write(
        "✅ Doctor/Admin sign-in "
        "and role-gated pages"
    )

    st.write(
        "✅ X-ray AI inference "
        "and human review workflow"
    )

    try:
        get_db().table(
            "audit_logs"
        ).select(
            "event_id"
        ).limit(1).execute()

        st.success(
            "Supabase Audit Log connection: available"
        )

    except Exception:
        st.error(
            "Supabase Audit Log connection: unavailable. "
            "Check Secrets, SQL permissions, and logs."
        )

    st.info(
        "The host manages physical infrastructure. "
        "Database access uses a high-privilege "
        "server secret. This is a teaching prototype, "
        "not a compliant clinical system."
    )


# ==========================================
# 12. UPLOAD X-RAY
# ==========================================

elif (
    page == "Upload X-ray"
    and role == "doctor"
):

    st.header("Upload X-ray")

    st.info(
        "Use only public, already de-identified "
        "PNG/JPEG images. This app does NOT remove "
        "text burned into image pixels."
    )

    uploaded = st.file_uploader(
        "PNG/JPEG image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded is not None:

        try:
            raw = uploaded.getvalue()

            image = read_image(raw)

            file_hash = hashlib.sha256(
                raw
            ).hexdigest()

            if (
                file_hash != st.session_state.image_hash
                or not st.session_state.image_id
                or st.session_state.xray_image is None
            ):

                clear_case()

                st.session_state.xray_image = image
                st.session_state.image_hash = file_hash
                st.session_state.image_id = new_id()

                audit(
                    "IMAGE_UPLOADED",
                    {
                        "image_id":
                            st.session_state.image_id,
                        "image_sha256": file_hash,
                        "size_bytes": len(raw)
                    }
                )

            st.image(
                st.session_state.xray_image,
                width=380
            )

            st.success(
                "Image ready for AI analysis"
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
# 13. AI ANALYSIS
# ==========================================

elif (
    page == "AI Analysis"
    and role == "doctor"
):

    st.header(
        "AI Chest X-ray Analysis"
    )

    if st.session_state.xray_image is None:
        st.info(
            "Please upload an image first."
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
                with st.spinner(
                    "Running model inference..."
                ):
                    scores = predict(
                        st.session_state.xray_image
                    )

                if not st.session_state.image_id:
                    raise ValueError(
                        "Missing image ID. "
                        "Re-upload the image."
                    )

                analysis_id = new_id()

                st.session_state.results = scores
                st.session_state.analysis_id = analysis_id
                st.session_state.review = None

                saved = audit(
                    "AI_INFERENCE_COMPLETED",
                    {
                        "image_id":
                            st.session_state.image_id,
                        "analysis_id": analysis_id,
                        "model": MODEL_ID,
                        "scores": scores
                    }
                )

                if saved:
                    st.success(
                        "AI analysis completed "
                        "and event saved to Supabase."
                    )

                else:
                    st.warning(
                        "AI analysis completed, "
                        "but the audit event was "
                        "NOT saved to Supabase."
                    )

            except Exception as exc:
                st.error(
                    "AI inference failed"
                )
                st.exception(exc)

        if st.session_state.results is not None:

            st.subheader(
                "Model Output Scores"
            )

            st.caption(
                "Scores are NOT calibrated disease "
                "probabilities or medical diagnoses."
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
# 14. DOCTOR REVIEW
# ==========================================

elif (
    page == "Doctor Review"
    and role == "doctor"
):

    st.header(
        "Doctor Review – 醫師審查"
    )

    st.caption(
        "The 'doctor' account is a demo role; "
        "medical qualifications are not verified."
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
                "Review notes / 修改原因 "
                "(demo text only)"
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

            elif (
                not st.session_state.image_id
                or not st.session_state.analysis_id
            ):
                st.error(
                    "Missing Image ID or Analysis ID. "
                    "Please upload and analyze again."
                )

            else:

                review = {
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

                st.session_state.review = review

                saved = audit(
                    "DOCTOR_REVIEW_COMPLETED",
                    review
                )

                if saved:
                    st.success(
                        "Review and audit event "
                        "saved to Supabase."
                    )

                else:
                    st.warning(
                        "Review saved in this session only; "
                        "Supabase audit insert FAILED."
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
# 15. MODEL CONTROL
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
        st.warning(
            "Admin view only. Model weight verification, "
            "version approvals and model switching "
            "are not implemented."
        )

    else:
        st.info(
            "Doctor: read-only model information."
        )


# ==========================================
# 16. PERSISTENT AUDIT LOG
# ==========================================

elif (
    page == "Audit Log"
    and role == "admin"
):

    st.header(
        "Persistent Audit Log – 稽核紀錄"
    )

    st.caption(
        "Reads saved events from Supabase "
        "(across browser sessions), "
        "not from Streamlit Session State."
    )

    if st.button("Refresh Audit Logs"):
        st.rerun()

    try:
        result = (
            get_db()
            .table("audit_logs")
            .select(
                "event_id,created_at,actor,role,"
                "action,image_id,analysis_id,details"
            )
            .order(
                "created_at",
                desc=True
            )
            .limit(500)
            .execute()
        )

        rows = result.data or []

        st.metric(
            "Persisted events (latest 500)",
            len(rows)
        )

        if rows:

            summary = []

            for event in rows:
                summary.append({
                    "created_at": event.get(
                        "created_at", ""
                    ),
                    "actor": event.get(
                        "actor", "legacy/unknown"
                    ),
                    "role": event.get(
                        "role", "unknown"
                    ),
                    "action": event.get(
                        "action", "UNKNOWN"
                    ),
                    "image_id": event.get(
                        "image_id"
                    ),
                    "analysis_id": event.get(
                        "analysis_id"
                    ),
                    "event_id": event.get(
                        "event_id"
                    )
                })

            st.dataframe(
                summary,
                hide_index=True,
                use_container_width=True
            )

            st.subheader(
                "Event Details"
            )

            for event in rows:

                title = (
                    f"{event.get('created_at', '')} | "
                    f"{event.get('action', 'UNKNOWN')}"
                )

                with st.expander(title):
                    st.json(event)

            st.download_button(
                "Download Persistent Audit Log (JSON)",
                data=json.dumps(
                    rows,
                    indent=2,
                    ensure_ascii=False,
                    default=str
                ),
                file_name="persistent_audit_log.json",
                mime="application/json"
            )

        else:
            st.info(
                "No saved events yet. "
                "Log in as Doctor and complete "
                "an upload, analysis and review."
            )

    except Exception:
        st.error(
            "Failed to read Supabase audit_logs. "
            "Check API key, permissions, database "
            "status and Cloud Logs."
        )
