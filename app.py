"""
app.py — Streamlit dashboard for Hand Gesture Recognition.

Run:
    streamlit run app.py

Requirements:
    pip install streamlit tensorflow pillow opencv-python-headless scikit-learn
"""

import os
import pickle

import numpy as np
import streamlit as st
from PIL import Image

from utils import preprocess_image, decode_prediction, DISPLAY_NAMES, GESTURE_EMOJI

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hand Gesture Recognition",
    page_icon="✋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

.main-header {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a3e 50%, #0f1a2e 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    border: 1px solid rgba(99,102,241,0.3);
    box-shadow: 0 0 40px rgba(99,102,241,0.15);
}
.main-header h1 { color: #e0e7ff; font-size: 2rem; margin: 0; letter-spacing: -0.5px; }
.main-header p  { color: #94a3b8; margin: 0.5rem 0 0; font-size: 0.95rem; }

.result-card {
    background: linear-gradient(135deg, #1e1b4b, #1e3a5f);
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(99,102,241,0.2);
    animation: fadeIn 0.4s ease;
}
@keyframes fadeIn {
    from { opacity:0; transform:translateY(10px); }
    to   { opacity:1; transform:translateY(0); }
}
.result-emoji { font-size: 4rem; line-height:1; margin-bottom:0.75rem; }
.result-name  { font-family:'Space Mono',monospace; font-size:1.6rem; font-weight:700; color:#e0e7ff; margin-bottom:0.25rem; }
.result-conf  { font-size:1.1rem; color:#6ee7b7; font-weight:600; }

.bar-outer { background:rgba(255,255,255,0.08); border-radius:8px; height:10px; margin-top:0.3rem; overflow:hidden; }
.bar-inner  { height:100%; border-radius:8px; background:linear-gradient(90deg,#6366f1,#06b6d4); }

.alt-row {
    display:flex; align-items:center; gap:0.75rem;
    padding:0.6rem 0; border-bottom:1px solid rgba(255,255,255,0.06);
    color:#cbd5e1; font-size:0.92rem;
}
.badge {
    display:inline-block; background:rgba(99,102,241,0.2);
    border:1px solid rgba(99,102,241,0.4); border-radius:20px;
    padding:2px 10px; font-size:0.75rem; color:#a5b4fc;
    font-family:'Space Mono',monospace; margin-left:auto;
}
.info-box {
    background:rgba(6,182,212,0.08); border-left:3px solid #06b6d4;
    border-radius:0 8px 8px 0; padding:0.75rem 1rem;
    color:#94a3b8; font-size:0.88rem; margin-bottom:1rem;
}
.stButton > button {
    background:linear-gradient(135deg,#6366f1,#06b6d4);
    color:white; border:none; border-radius:10px;
    font-family:'Space Mono',monospace; font-weight:700;
    letter-spacing:0.5px; padding:0.6rem 1.5rem;
    width:100%; transition:opacity 0.2s;
}
.stButton > button:hover { opacity:0.88; }
.sidebar-section {
    background:rgba(99,102,241,0.08); border-radius:10px;
    padding:1rem; margin-bottom:1rem;
    border:1px solid rgba(99,102,241,0.2);
}
</style>
""", unsafe_allow_html=True)


# ─── Compat helper ────────────────────────────────────────────────────────────

def show_image(img, caption=None):
    """
    st.image wrapper that works on both old (<1.20) and new (>=1.20) Streamlit.
    Old versions use `use_column_width`; new versions use `use_container_width`.
    """
    try:
        st.image(img, caption=caption, use_container_width=True)
    except TypeError:
        st.image(img, caption=caption, use_column_width=True)


# ─── Model loading ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def load_model_and_encoders(model_dir: str):
    import tensorflow as tf
    model_path   = os.path.join(model_dir, "gesture_model.keras")
    encoder_path = os.path.join(model_dir, "label_encoder.pkl")

    if not os.path.exists(model_path):
        return None, None, f"Model not found: {model_path}"
    if not os.path.exists(encoder_path):
        return None, None, f"Encoder not found: {encoder_path}"

    try:
        model = tf.keras.models.load_model(model_path)
        with open(encoder_path, "rb") as f:
            label_encoder = pickle.load(f)
        return model, label_encoder, None
    except Exception as e:
        return None, None, str(e)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    model_dir = st.text_input(
        "Model directory",
        value=r"A:\SCT_ML_4\models",
        help="Folder containing gesture_model.keras and label_encoder.pkl",
    )
    st.markdown("---")
    st.markdown("### 🔬 Prediction settings")
    top_k          = st.slider("Top-K results",            min_value=1, max_value=5,   value=3)
    conf_threshold = st.slider("Confidence threshold (%)", min_value=0, max_value=100, value=30)

    st.markdown("---")
    gesture_rows = "".join(
        f"{GESTURE_EMOJI.get(k,'🖐️')} {DISPLAY_NAMES.get(k,k)}<br>"
        for k in [
            "01_palm","02_I","03_fist","04_fist_moved","05_thumb",
            "06_index","07_ok","08_palm_moved","09_c","10_down",
        ]
    )
    st.markdown(
        f'<div class="sidebar-section"><b>Supported gestures</b><br><br>{gesture_rows}</div>',
        unsafe_allow_html=True,
    )


# ─── Header ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
  <h1>✋ Hand Gesture Recognition</h1>
  <p>Upload a hand image and the model will identify the gesture in real time.</p>
</div>
""", unsafe_allow_html=True)


# ─── Load model ───────────────────────────────────────────────────────────────

model, label_encoder, load_error = load_model_and_encoders(model_dir)

if load_error:
    st.error(f"❌ Could not load model: {load_error}")
    st.info("Train the model first (`python train.py`), then restart this app.")
    st.stop()

st.success("✅ Model loaded successfully")


# ─── Single image upload & predict ────────────────────────────────────────────

col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("#### 📤 Upload image")
    st.markdown(
        '<div class="info-box">Upload a photo of your hand. '
        'Works best with a plain background and good lighting.</div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Choose an image…",
        type=["jpg","jpeg","png","bmp","webp"],
        label_visibility="collapsed",
    )
    if uploaded:
        pil_img = Image.open(uploaded)
        show_image(pil_img, caption="Uploaded image")
        predict_btn = st.button("🔍 Recognise Gesture", use_container_width=True)
    else:
        predict_btn = False
        st.markdown(
            "<p style='color:#64748b;font-size:0.9rem;margin-top:1rem;'>"
            "No image uploaded yet.</p>",
            unsafe_allow_html=True,
        )

with col_result:
    st.markdown("#### 🎯 Prediction")

    if uploaded and predict_btn:
        with st.spinner("Analysing…"):
            try:
                tensor  = preprocess_image(pil_img)           # (1,128,128,3) uint8
                probs   = model.predict(tensor, verbose=0)[0]  # (num_classes,)
                results = decode_prediction(probs, label_encoder, top_k=top_k)
                top     = results[0]

                if top["confidence"] * 100 < conf_threshold:
                    st.warning(
                        f"⚠️ Best match **{top['display_name']}** has low confidence "
                        f"({top['confidence_pct']} < {conf_threshold}%). "
                        "Try a clearer image."
                    )

                # Top result card
                st.markdown(f"""
                <div class="result-card">
                  <div class="result-emoji">{top['emoji']}</div>
                  <div class="result-name">{top['display_name']}</div>
                  <div class="result-conf">{top['confidence_pct']} confidence</div>
                </div>
                """, unsafe_allow_html=True)

                # Runner-up bars
                if len(results) > 1:
                    st.markdown(
                        "<p style='color:#64748b;font-size:0.85rem;margin:1.2rem 0 0.3rem;'>"
                        "Other possibilities</p>",
                        unsafe_allow_html=True,
                    )
                    for r in results[1:]:
                        bar_w = int(r["confidence"] * 100)
                        st.markdown(f"""
                        <div class="alt-row">
                          <span>{r['emoji']}</span>
                          <span>{r['display_name']}</span>
                          <div style="flex:1">
                            <div class="bar-outer">
                              <div class="bar-inner" style="width:{bar_w}%"></div>
                            </div>
                          </div>
                          <span class="badge">{r['confidence_pct']}</span>
                        </div>
                        """, unsafe_allow_html=True)

                # Full probability chart
                with st.expander("📊 Full probability distribution"):
                    all_results = decode_prediction(probs, label_encoder, top_k=len(probs))
                    names   = [r["display_name"] for r in all_results]
                    confs   = [r["confidence"]    for r in all_results]
                    colours = ["#6366f1" if i == 0 else "#334155" for i in range(len(names))]

                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(6, 3.5))
                    fig.patch.set_facecolor("#0f0f1a")
                    ax.set_facecolor("#0f0f1a")
                    ax.barh(names[::-1], confs[::-1], color=colours[::-1], height=0.6)
                    ax.set_xlim(0, 1)
                    ax.set_xlabel("Confidence", color="#94a3b8", fontsize=9)
                    ax.tick_params(colors="#94a3b8", labelsize=8)
                    for spine in ax.spines.values():
                        spine.set_edgecolor("#334155")
                    for i, conf in enumerate(confs[::-1]):
                        ax.text(conf + 0.01, i, f"{conf*100:.1f}%",
                                va='center', color="#e0e7ff", fontsize=7)
                    plt.tight_layout(pad=1)
                    st.pyplot(fig)
                    plt.close(fig)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.exception(e)

    elif not uploaded:
        st.markdown(
            "<div style='color:#475569;padding:3rem 0;text-align:center;'>"
            "Upload an image on the left to see predictions here.</div>",
            unsafe_allow_html=True,
        )


# ─── Batch mode ───────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("#### 📦 Batch prediction")

batch_files = st.file_uploader(
    "Upload multiple images for batch prediction",
    type=["jpg","jpeg","png","bmp","webp"],
    accept_multiple_files=True,
    key="batch",
)

if batch_files and st.button("🚀 Run batch prediction"):
    n_cols = min(len(batch_files), 4)
    cols   = st.columns(n_cols)
    for idx, f in enumerate(batch_files):
        with cols[idx % n_cols]:
            try:
                img    = Image.open(f)
                tensor = preprocess_image(img)
                probs  = model.predict(tensor, verbose=0)[0]
                top    = decode_prediction(probs, label_encoder, top_k=1)[0]

                show_image(img)
                st.markdown(
                    f"<div style='text-align:center;color:#e0e7ff;font-size:0.85rem;'>"
                    f"{top['emoji']} <b>{top['display_name']}</b><br>"
                    f"<span style='color:#6ee7b7'>{top['confidence_pct']}</span></div>",
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"{f.name}: {e}")


# ─── Footer ───────────────────────────────────────────────────────────────────

st.markdown("""
<div style='text-align:center;color:#334155;font-size:0.78rem;
            margin-top:3rem;font-family:Space Mono,monospace;'>
  Hand Gesture Recognition · MobileNetV2 Transfer Learning · 10 Gestures
</div>
""", unsafe_allow_html=True)