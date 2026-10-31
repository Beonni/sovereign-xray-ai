
"""Sovereign X-ray AI v5 — teaching/demo application only."""
import hashlib
import hmac
import io
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from supabase import create_client

st.set_page_config(
    page_title="Sovereign X-ray AI",
    page_icon="🏥",
    layout="wide"
)

MODEL_ID = "densenet121-res224-all"
MAX_FILE_SIZE = 10 * 1024 * 1024


# ==========================================
# 1. UTILITIES
# ==========================================

def timestamp():
    return datetime.now(timezone.utc).isoformat()


def uid():
    return str(uuid.uuid4())


def init_state():
    defaults = dict(
        user=None,
        role=None,
        xray_image=None,
        image_hash=None,
        image_id=None,
        results=None,
        analysis_id=None,
        review=None,
        audit_logs=[],
        last_audit_error=None,
        failed_logins=0
    )

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
        "review"
    ):
        st.session_state[key] = None


# ==========================================
# 2. SUPABASE CONNECTION
# ==========================================

@st.cache_resource(show_spinner=False)
def connect_db(url, key):
    return create_client(url, key)


def db():
    url = str(
        st.secrets.get("SUPABASE_URL", "")
    ).strip()

    key = str(
        st.secrets.get("SUPABASE_SECRET_KEY", "")
    ).strip()

    if not url.startswith("https://") or not key:
        raise RuntimeError(
            "Configure SUPABASE_URL and "
            "SUPABASE_SECRET_KEY in Streamlit Secrets"
        )

    return connect_db(url, key)


# ==========================================
# 3. PERSISTENT AUDIT LOG
# ==========================================

def audit(action, details=None):
    details = details or {}

    record = dict(
        event_id=uid(),
        created_at=timestamp(),
        actor=st.session_state.user or "anonymous",
        role=st.session_state.role or "anonymous",
        action=action,
        image_id=details.get("image_id"),
        analysis_id=details.get("analysis_id"),
        details=details
    )

    try:
        db().table("audit_logs").insert(
            record
        ).execute()

        st.session_state.audit_logs.append({
            **record,
            "sync_status": "SAVED"
        })

        st.session_state.last_audit_error = None
        return True

    except Exception:
        st.session_state.audit_logs.append({
            **record,
            "sync_status": "NOT_SAVED"
        })

        st.session_state.last_audit_error = (
            "Audit event NOT saved in Supabase. "
            "Check database connection and permissions."
        )

        return False


# ==========================================
# 4. AUTHENTICATION
# ==========================================

def authenticate(username, password):
    if username not in ("doctor", "admin") or not password:
        return False

    try:
        entry = st.secrets[username]

        if not hmac.compare_digest(
            str(entry["username"]),
            username
        ):
            return False

        salt = bytes.fromhex(
            str(entry["salt"])
        )

        expected = bytes.fromhex(
            str(entry["password_hash"])
        )

        iterations = int(entry["iterations"])

        if (
            iterations < 100000
            or len(expected) != 32
            or not salt
        ):
            return False

        candidate = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode(),
            salt,
            iterations
        )

        return hmac.compare_digest(
            candidate,
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
# 5. IMAGE VALIDATION
# ==========================================

def read_image(data):
    if not data or len(data) > MAX_FILE_SIZE:
        raise ValueError(
            "Image must be nonempty and 10MB or smaller"
        )

    with Image.open(io.BytesIO(data)) as source:
        if source.format not in ("JPEG", "PNG"):
            raise ValueError(
                "Only PNG or JPEG accepted"
            )

        if max(source.size) > 5000:
            raise ValueError("Image is too large")

        result = source.convert("L")
        result.load()

    return result


# ==========================================
# 6. AI MODEL & WEIGHTS
# ==========================================

@st.cache_resource(
    show_spinner="Loading X-ray model..."
)
def loaded_model():
    import torchxrayvision as xrv

    model = xrv.models.DenseNet(
        weights=MODEL_ID
    )

    model.eval()
    return model


@st.cache_resource(
    show_spinner="Locating pretrained model weights..."
)
def weights_path():
    import torchxrayvision as xrv

    return Path(
        xrv.models.get_weights(MODEL_ID)
    )


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for part in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            digest.update(part)

    return digest.hexdigest()


def actual_fingerprint():
    path = weights_path()

    if not path.is_file():
        raise RuntimeError(
            "Pretrained weights file is missing"
        )

    return sha256_file(path)


# ==========================================
# 7. MODEL REGISTRY
# ==========================================

def registry():
    result = (
        db()
        .table("model_registry")
        .select("*")
        .eq("model_id", MODEL_ID)
        .limit(1)
        .execute()
    )

    if not result.data:
        raise RuntimeError(
            "Model not registered in "
            "Supabase model_registry"
        )

    return result.data[0]


def change_registry(payload):
    result = (
        db()
        .table("model_registry")
        .update(payload)
        .eq("model_id", MODEL_ID)
        .execute()
    )

    if not result.data:
        raise RuntimeError(
            "Model registry update "
            "did not return a record"
        )

    return result.data[0]


def checked_model():
    item = registry()

    if item.get("status") != "APPROVED":
        raise PermissionError(
            "Model not approved. "
            "Ask Admin to inspect and approve it."
        )

    approved_hash = item.get("weights_sha256")

    if not approved_hash:
        raise PermissionError(
            "Model does not have "
            "a registered SHA-256 hash"
        )

    path = weights_path()
    before = sha256_file(path)

    if not hmac.compare_digest(
        before,
        approved_hash
    ):
        raise PermissionError(
            "Weight SHA-256 mismatch. "
            "Inference blocked."
        )

    model = loaded_model()

    after = sha256_file(path)

    if not hmac.compare_digest(
        after,
        approved_hash
    ):
        raise PermissionError(
            "Weights changed during loading. "
            "Inference blocked."
        )

    return model, before


# ==========================================
# 8. AI INFERENCE
# ==========================================

def predict(image, model):
    import torchxrayvision as xrv

    img = np.asarray(
        image.convert("L"),
        dtype=np.float32
    )

    img = xrv.datasets.normalize(
        img, 255
    )[None, :, :]

    img = xrv.datasets.XRayCenterCrop()(img)
    img = xrv.datasets.XRayResizer(224)(img)

    tensor = torch.from_numpy(
        np.ascontiguousarray(img)
    ).unsqueeze(0).float()

    with torch.inference_mode():
        output = model(tensor)[0].cpu().numpy()

    scores = [
        {
            "finding": label,
            "score": float(value)
        }
        for label, value in zip(
            model.pathologies,
            output
        )
        if np.isfinite(value)
    ]

    return sorted(
        scores,
        key=lambda x: x["score"],
        reverse=True
    )


# ==========================================
# 9. APPLICATION INITIALIZATION
# ==========================================

init_state()

st.title(
    "🏥 Sovereign AI Medical Imaging System"
)

st.caption(
    "基於主權 AI 之胸部 X 光影像輔助分析系統 "
    "· Model Governance v5"
)

st.warning(
    "Research demonstration only; "
    "not for clinical diagnosis. "
    "Use public, de-identified test images only."
)


# ==========================================
# 10. LOGIN PAGE
# ==========================================

if st.session_state.user is None:
    st.subheader("Sign in – 身分驗證")

    with st.form("login"):
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
        if authenticate(username, password):
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
                {"attempted_username": username[:50]}
            )

            st.error(
                "Incorrect username or password"
            )

    st.stop()


# ==========================================
# 11. ROLE-BASED NAVIGATION
# ==========================================

role = st.session_state.role

if role not in ("doctor", "admin"):
    st.error("Unauthorized role")
    st.stop()

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

if st.session_state.last_audit_error:
    st.error(
        st.session_state.last_audit_error
    )


# ==========================================
# 12. DASHBOARD
# ==========================================

if page == "Dashboard":
    st.header("System Dashboard")

    a, b, c = st.columns(3)

    a.metric("Role", role.title())
    b.metric("Model", "DenseNet121")
    c.metric(
        "Session events",
        len(st.session_state.audit_logs)
    )

    try:
        db().table(
            "audit_logs"
        ).select(
            "event_id"
        ).limit(1).execute()

        model_item = registry()

        st.success(
            "Supabase Audit Log connection: available"
        )

        st.write(
            "Model approval status:",
            model_item.get("status", "UNKNOWN")
        )

        st.write(
            "Registered model SHA-256:",
            model_item.get("weights_sha256")
            or "Not registered"
        )

    except Exception as exc:
        st.error(
            "Database/Model Registry unavailable: "
            f"{type(exc).__name__}"
        )

    st.info(
        "Third-party cloud infrastructure; "
        "model approval is a prototype control, "
        "not clinical certification."
    )


# ==========================================
# 13. UPLOAD X-RAY
# ==========================================

elif page == "Upload X-ray" and role == "doctor":
    st.header("Upload X-ray")

    st.info(
        "Only public, already de-identified "
        "PNG/JPEG images. Embedded text in "
        "pixels is NOT removed."
    )

    uploaded = st.file_uploader(
        "Choose X-ray image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded is not None:
        try:
            raw = uploaded.getvalue()
            image = read_image(raw)

            digest = hashlib.sha256(
                raw
            ).hexdigest()

            if (
                digest != st.session_state.image_hash
                or not st.session_state.image_id
                or st.session_state.xray_image is None
            ):
                clear_case()

                st.session_state.xray_image = image
                st.session_state.image_hash = digest
                st.session_state.image_id = uid()

                audit(
                    "IMAGE_UPLOADED",
                    {
                        "image_id":
                            st.session_state.image_id,
                        "image_sha256": digest,
                        "size_bytes": len(raw)
                    }
                )

            st.image(
                st.session_state.xray_image,
                width=380
            )

            st.success("Image ready")

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
            width=350
        )

        st.write(
            "Current image ID:",
            st.session_state.image_id
        )


# ==========================================
# 14. AI ANALYSIS
# ==========================================

elif page == "AI Analysis" and role == "doctor":
    st.header("AI Chest X-ray Analysis")

    if st.session_state.xray_image is None:
        st.info(
            "Please upload an image first"
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

        try:
            state = registry()

            st.write(
                "Model status:",
                state.get("status")
            )

            ready = (
                state.get("status") == "APPROVED"
                and bool(state.get("weights_sha256"))
            )

        except Exception:
            ready = False
            st.error(
                "Cannot verify model approval "
                "in Supabase"
            )

        if not ready:
            st.warning(
                "Inference disabled until Admin "
                "verifies and approves the checkpoint"
            )

        if st.button(
            "Analyze X-ray with AI",
            type="primary",
            disabled=not ready
        ):
            try:
                with st.spinner(
                    "Verifying model integrity "
                    "and running AI..."
                ):
                    model, checkpoint_hash = checked_model()

                    scores = predict(
                        st.session_state.xray_image,
                        model
                    )

                analysis_id = uid()

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
                        "weights_sha256": checkpoint_hash,
                        "scores": scores
                    }
                )

                if saved:
                    st.success(
                        "AI inference completed; "
                        "audit saved to Supabase"
                    )
                else:
                    st.warning(
                        "Inference completed, "
                        "but audit was NOT saved "
                        "to Supabase"
                    )

            except Exception as exc:
                st.error(
                    f"Inference blocked or failed: {exc}"
                )

        if st.session_state.results is not None:
            st.subheader("Model output scores")

            st.caption(
                "Scores are not calibrated disease "
                "probabilities or diagnoses."
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
# 15. DOCTOR REVIEW
# ==========================================

elif page == "Doctor Review" and role == "doctor":
    st.header(
        "Doctor Review – 醫師審查"
    )

    st.caption(
        "Doctor is a demo role; medical "
        "qualifications have not been verified."
    )

    if st.session_state.results is None:
        st.info(
            "Complete AI Analysis first"
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
            "Analysis ID:",
            st.session_state.analysis_id
        )

        st.dataframe(
            st.session_state.results,
            hide_index=True,
            use_container_width=True
        )

        with st.form("review"):
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
                st.error("Review notes required")

            elif (
                not st.session_state.image_id
                or not st.session_state.analysis_id
            ):
                st.error(
                    "Missing image or analysis ID"
                )

            else:
                item = {
                    "review_id": uid(),
                    "image_id": st.session_state.image_id,
                    "analysis_id": st.session_state.analysis_id,
                    "reviewer": st.session_state.user,
                    "decision": decision,
                    "notes": notes.strip(),
                    "model": MODEL_ID,
                    "timestamp": timestamp()
                }

                st.session_state.review = item

                if audit(
                    "DOCTOR_REVIEW_COMPLETED",
                    item
                ):
                    st.success(
                        "Review audit saved to Supabase"
                    )
                else:
                    st.warning(
                        "Review exists only in session; "
                        "Supabase save failed"
                    )

        if st.session_state.review is not None:
            st.subheader("Latest Review")
            st.json(st.session_state.review)

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
# 16. MODEL CONTROL
# ==========================================

elif page == "Model Control":
    st.header(
        "Model Control – 模型管理"
    )

    st.write("Model:", MODEL_ID)
    st.write(
        "Framework: PyTorch / TorchXRayVision"
    )

    st.caption(
        "SHA-256 identifies checkpoint file bytes, "
        "not clinical performance certification."
    )

    try:
        item = registry()

        st.write(
            "Registry status:",
            item.get("status")
        )

        st.write(
            "Stored SHA-256:",
            item.get("weights_sha256")
            or "Not registered"
        )

        st.write(
            "Approved by:",
            item.get("approved_by") or "—"
        )

        st.write(
            "Approved at:",
            item.get("approved_at") or "—"
        )

    except Exception as exc:
        st.error(
            f"Model registry unavailable: {exc}"
        )
        st.stop()

    if role == "admin":

        # Step A: Load model and compute hash
        if st.button(
            "Load model and compute checkpoint SHA-256"
        ):
            try:
                with st.spinner(
                    "Loading checkpoint and hashing file..."
                ):
                    loaded_model()
                    actual = actual_fingerprint()

                st.session_state.current_weights_hash = actual

                st.success(
                    "Checkpoint fingerprint calculated"
                )

            except Exception as exc:
                st.error(
                    f"Checkpoint unavailable: {exc}"
                )

        current_hash = st.session_state.get(
            "current_weights_hash"
        )

        if current_hash:
            st.write(
                "Current checkpoint SHA-256:",
                current_hash
            )

            stored = item.get("weights_sha256")

            st.write(
                "Matches registered hash:",
                bool(
                    stored and hmac.compare_digest(
                        stored,
                        current_hash
                    )
                )
            )

            # Step B: Register actual hash
            if item.get("status") != "APPROVED":

                if st.button(
                    "Register current checkpoint SHA-256"
                ):
                    try:
                        change_registry({
                            "weights_sha256": current_hash,
                            "status": "PENDING",
                            "approved_by": None,
                            "approved_at": None
                        })

                        audit(
                            "MODEL_HASH_REGISTERED",
                            {
                                "model": MODEL_ID,
                                "weights_sha256": current_hash
                            }
                        )

                        st.success(
                            "Fingerprint registered as PENDING"
                        )

                        st.rerun()

                    except Exception as exc:
                        st.error(
                            f"Registration failed: {exc}"
                        )

        # Step C: Approve model
        if item.get("status") != "APPROVED":

            if st.button(
                "Approve verified model",
                type="primary"
            ):
                try:
                    with st.spinner(
                        "Rechecking checkpoint SHA-256..."
                    ):
                        loaded_model()
                        measured = actual_fingerprint()

                    approved = item.get("weights_sha256")

                    if (
                        not approved
                        or not hmac.compare_digest(
                            approved,
                            measured
                        )
                    ):
                        raise ValueError(
                            "Current checkpoint SHA-256 "
                            "does not match registered hash"
                        )

                    change_registry({
                        "status": "APPROVED",
                        "approved_by": st.session_state.user,
                        "approved_at": timestamp()
                    })

                    audit(
                        "MODEL_APPROVED",
                        {
                            "model": MODEL_ID,
                            "weights_sha256": measured
                        }
                    )

                    st.success(
                        "Model approved for demonstration"
                    )

                    st.rerun()

                except Exception as exc:
                    st.error(
                        f"Approval denied: {exc}"
                    )

        # Step D: Revoke approval
        else:

            if st.button("Revoke model approval"):
                try:
                    change_registry({
                        "status": "REJECTED",
                        "approved_by": None,
                        "approved_at": None
                    })

                    audit(
                        "MODEL_APPROVAL_REVOKED",
                        {"model": MODEL_ID}
                    )

                    st.success(
                        "Approval revoked; "
                        "future inference will be blocked"
                    )

                    st.rerun()

                except Exception as exc:
                    st.error(
                        f"Revocation failed: {exc}"
                    )

    else:
        st.info(
            "Doctor has read-only model information"
        )


# ==========================================
# 17. PERSISTENT AUDIT LOG
# ==========================================

elif page == "Audit Log" and role == "admin":
    st.header(
        "Persistent Audit Log – 稽核紀錄"
    )

    st.caption(
        "Supabase shared events (last 500), "
        "not session-only logs"
    )

    if st.button("Refresh logs"):
        st.rerun()

    try:
        response = (
            db()
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

        rows = response.data or []

        st.metric(
            "Persisted events (latest 500)",
            len(rows)
        )

        if rows:
            summary = [
                {
                    "created_at": r.get("created_at", ""),
                    "actor": r.get(
                        "actor",
                        "legacy/unknown"
                    ),
                    "role": r.get(
                        "role",
                        "unknown"
                    ),
                    "action": r.get(
                        "action",
                        "UNKNOWN"
                    ),
                    "event_id": r.get(
                        "event_id",
                        ""
                    )
                }
                for r in rows
            ]

            st.dataframe(
                summary,
                hide_index=True,
                use_container_width=True
            )

            for row in rows:
                title = (
                    f"{row.get('created_at', '')} | "
                    f"{row.get('action', 'UNKNOWN')}"
                )

                with st.expander(title):
                    st.json(row)

            st.download_button(
                "Download Audit Log JSON",
                json.dumps(
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
                "No persisted events yet"
            )

    except Exception as exc:
        st.error(
            "Could not read Supabase audit log: "
            f"{type(exc).__name__}"
        )
