
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Sovereign AI Medical System",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Sovereign AI Medical Imaging System")
st.caption("基於主權 AI 之胸部 X 光影像輔助分析系統")

st.warning(
    "This system is for educational demonstration only."
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
    st.success("Website is running successfully!")

elif page == "Upload X-ray":
    st.header("Upload Chest X-ray")

    file = st.file_uploader(
        "Choose X-ray image",
        type=["png", "jpg", "jpeg"]
    )

    if file is not None:
        image = Image.open(file)
        st.image(image, caption="Uploaded X-ray", width=400)

elif page == "AI Analysis":
    st.header("AI Analysis")
    st.info("AI model will be integrated in the next stage.")

elif page == "Doctor Review":
    st.header("Doctor Review")
    st.info("Doctor review module is under development.")

elif page == "Model Control":
    st.header("Model Management")
    st.info("Model version control will be added.")

elif page == "Audit Log":
    st.header("Audit Log")
    st.info("Audit log functionality will be added.")
