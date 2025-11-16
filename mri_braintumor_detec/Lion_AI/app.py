import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import pandas as pd

# ======================================================
# 🧠 โหลดโมเดล YOLOv8
# ======================================================
MODEL_PATH = 'YOLOv8_Results/mri_tumor/weights/best.pt'
model = YOLO(MODEL_PATH)

# ======================================================
# 🎨 Custom CSS - Dark Theme (Neural Grid & Glow)
# ======================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

#MainMenu, footer, header {visibility: hidden;}

.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    font-family: 'Space Grotesk', sans-serif;
    position: relative;
}

/* 🔷 Grid Background */
.stApp::before {
    content: '';
    position: fixed; top: 0; left: 0;
    width: 100%; height: 100%;
    background-image: 
        repeating-linear-gradient(0deg, transparent, transparent 49px, rgba(99, 102, 241, 0.1) 49px, rgba(99, 102, 241, 0.1) 50px),
        repeating-linear-gradient(90deg, transparent, transparent 49px, rgba(168, 85, 247, 0.1) 49px, rgba(168, 85, 247, 0.1) 50px);
    filter: drop-shadow(0 0 1px rgba(168, 85, 247, 0.3)) drop-shadow(0 0 1px rgba(99, 102, 241, 0.3));
    pointer-events: none; z-index: 1;
}

/* 🟣 Glowing Orbs */
.stApp::after {
    content: '';
    position: fixed; top: 0; left: 0;
    width: 100%; height: 100%;
    background:
        radial-gradient(circle at 20% 30%, rgba(168, 85, 247, 0.15) 0%, transparent 25%),
        radial-gradient(circle at 80% 70%, rgba(99, 102, 241, 0.12) 0%, transparent 25%),
        radial-gradient(circle at 50% 50%, rgba(139, 92, 246, 0.1) 0%, transparent 30%);
    pointer-events: none; z-index: 1;
}

.stApp > div { position: relative; z-index: 2; }

/* 🧩 Title & Subtitle */
.main-title {
    font-size: 2.5rem; font-weight: 700;
    color: #f1f5f9; text-align: center;
    margin-bottom: 0.5rem;
    text-shadow: 0 0 20px rgba(168, 85, 247, 0.5);
}
.subtitle {
    font-size: 1rem; color: #a5b4fc;
    text-align: center; margin-bottom: 2rem;
}

/* 🖼️ Image Container */
.img-container {
    border-radius: 12px; overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,0,0,0.6);
    border: 1px solid rgba(168, 85, 247, 0.3);
    margin-bottom: 1rem;
}

/* 📊 Metrics */
.stats-badge {
    display: inline-block;
    background: rgba(168, 85, 247, 0.2);
    color: #c4b5fd; padding: 0.625rem 1.25rem;
    border-radius: 20px; font-size: 0.9rem;
    font-weight: 600; margin: 0.5rem;
    border: 1px solid rgba(168, 85, 247, 0.4);
}

[data-testid="stMetricValue"] {
    color: #c4b5fd !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-weight: 600 !important;
}

/* ✨ Text Styling */
.stMarkdown, p, label, h1, h2, h3, h4, h5, h6 { color: #cbd5e1 !important; }
strong { color: #a5b4fc !important; }

/* Divider */
hr {
    border: none; height: 2px;
    background: linear-gradient(90deg, transparent 0%, rgba(168, 85, 247, 0.5) 50%, transparent 100%);
    margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# 🧠 UI Layout
# ======================================================

st.markdown('<h1 class="main-title">🧠 Brain Tumor Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced MRI Analysis • Real-time Detection • Powered by YOLOv8</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file:
    with st.spinner("🔍 Analyzing neural patterns..."):
        image = Image.open(uploaded_file)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            image.save(temp_file.name)
            results = model.predict(source=temp_file.name, conf=0.25, save=False, show=False)

        result_image = results[0].plot()
        boxes = results[0].boxes.data.cpu().numpy()

    #st.markdown("---")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("**⚡ Original Scan**")
        #st.markdown('<div class="img-container">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("**🎯 Detection Result**")
        #st.markdown('<div class="img-container">', unsafe_allow_html=True)
        st.image(result_image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    #st.markdown("---")

    if len(boxes) > 0:
        st.markdown("### 📊 Neural Analysis Report")

        avg_conf = boxes[:, 4].mean() * 100
        col_a, col_b, col_c = st.columns(3)

        col_a.metric("Objects Detected", len(boxes))
        col_b.metric("Avg Confidence", f"{avg_conf:.1f}%")
        col_c.metric("Neural Engine", "YOLOv8")

        with st.expander("🔬 View Detailed Neural Analysis"):
            class_names = results[0].names if hasattr(results[0], 'names') else {}
            df_data = []

            for i, box in enumerate(boxes):
                x1, y1, x2, y2, conf = box[:5]
                cls = int(box[5]) if len(box) > 5 else 0
                class_name = class_names.get(cls, f"Class {cls}")
                conf_pct = conf * 100

                st.markdown(
                    f'<span class="stats-badge">⚡ Detection {i+1} • {class_name} • Confidence: {conf_pct:.1f}%</span>',
                    unsafe_allow_html=True
                )

                df_data.append({
                    '🔍 Detection #': i+1,
                    '🏷️ Class': class_name,
                    '📍 X1 (Left)': f"{x1:.1f}",
                    '📍 Y1 (Top)': f"{y1:.1f}",
                    '📍 X2 (Right)': f"{x2:.1f}",
                    '📍 Y2 (Bottom)': f"{y2:.1f}",
                    '🎯 Confidence': f"{conf_pct:.2f}%"
                })

            st.markdown("**Raw Neural Data:**")
            st.dataframe(pd.DataFrame(df_data), use_container_width=True, hide_index=True)
    else:
        st.info("✨ No abnormalities detected • Neural scan complete")

else:
    #st.markdown("---")
    st.info("👆 Upload an MRI scan to begin neural analysis")

# ======================================================
# ⚡ Footer
# ======================================================
#st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #64748b; font-size: 0.875rem;'>⚡ POWERED BY YOLOv8 • BUILT WITH STREAMLIT</p>",
    unsafe_allow_html=True
)
