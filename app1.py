"""
AgriBot — Rice Disease Detector (Streamlit Cloud edition)
===========================================================
Web-friendly rewrite of AgriBot_7.py.

The original AgriBot_7.py is a Tkinter DESKTOP app that also drives an
Arduino-based 4WD RC car over a serial/COM port and speaks alerts through
the local machine's speakers (pyttsx3). None of that is possible on
Streamlit Community Cloud:
  - No access to a user's serial/USB ports (Arduino control removed).
  - No persistent OpenCV VideoCapture loop against a "server webcam"
    (replaced with st.camera_input, which grabs a snapshot from the
    *visitor's* browser camera — the only camera access a hosted web
    app can get).
  - No server-side audio playback (pyttsx3 removed; replaced with an
    optional gTTS clip played back in-browser via st.audio).

Everything else — YOLO rice-disease detection, bounding boxes, and the
remedy advisor panel — is preserved.
"""

import io
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ─────────────────────────────────────────────────────────────
# CONSTANTS  (kept identical to the original AgriBot_7.py)
# ─────────────────────────────────────────────────────────────
CLASS_NAMES = ["false_smut", "sheath_blight", "sheath_rot", "stem_borer"]

REMEDIES = {
    "false_smut":    "Apply fungicide (Propiconazole) at booting & heading stage. Use certified disease-free seeds.",
    "sheath_blight": "Apply Validamycin 3L @ 2mL/L. Reduce plant density and avoid over-irrigation.",
    "sheath_rot":    "Use certified seeds. Spray Carbendazim 50WP @ 1g/L at boot leaf stage.",
    "stem_borer":    "Apply Cartap Hydrochloride 4G @ 18kg/ha. Release Trichogramma parasitoids.",
}

DISEASE_COLORS = {
    "false_smut":    "#F59E0B",
    "sheath_blight": "#EF4444",
    "sheath_rot":    "#8B5CF6",
    "stem_borer":    "#10B981",
}

MODEL_PATH = "best.pt"

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG + THEME
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgriBot — Rice Disease Detector",
    page_icon="🌾",
    layout="wide",
)

st.markdown(
    """
    <style>
    html, body, [class*="css"]  { font-family: 'Courier New', monospace; }
    .stApp { background-color: #0F172A; color: #E2E8F0; }
    section[data-testid="stSidebar"] { background-color: #020617; }
    .badge {
        padding: 10px 14px; border-radius: 6px; font-weight: bold;
        text-align: center; font-size: 15px; margin-bottom: 10px;
    }
    .remedy-box {
        background-color: #1E293B; border-radius: 8px; padding: 16px;
        border-left: 4px solid #34D399;
    }
    .remedy-step {
        color: #94A3B8; font-size: 14px; margin: 6px 0;
    }
    .step-num { color: #34D399; font-weight: bold; margin-right: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# MODEL LOADING (cached — runs once per server process)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading YOLO model…")
def load_model():
    from ultralytics import YOLO
    return YOLO(MODEL_PATH)


def hex_to_bgr(hx: str):
    hx = hx.lstrip("#")
    r, g, b = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
    return (b, g, r)


def run_detection(image_bgr: np.ndarray, model, conf: float):
    """Run YOLO on a BGR numpy image, return (annotated_bgr, detections)."""
    results = model(image_bgr, verbose=False, conf=conf)
    dets = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls = int(box.cls[0])
            det_conf = float(box.conf[0])
            lbl = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else r.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            dets.append((lbl, det_conf, x1, y1, x2, y2))

    annotated = image_bgr.copy()
    for (lbl, det_conf, x1, y1, x2, y2) in dets:
        bgr = hex_to_bgr(DISEASE_COLORS.get(lbl, "#FFFFFF"))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr, 2)
        txt = f"{lbl} {det_conf:.0%}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 8, y1), bgr, -1)
        cv2.putText(annotated, txt, (x1 + 4, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return annotated, dets


def render_remedy_panel(dets):
    if not dets:
        st.markdown(
            '<div class="badge" style="background:#1E293B;color:#475569;">'
            'NO DISEASE DETECTED</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="remedy-box" style="text-align:center;">'
            '<div style="color:#34D399;font-size:18px;font-weight:bold;">✅ Healthy</div>'
            '<div style="color:#475569;font-size:13px;margin-top:8px;">'
            'No disease detected.<br>Aim the camera at rice plants<br>'
            'or upload an image to scan.</div></div>',
            unsafe_allow_html=True)
        return None

    top = max(dets, key=lambda x: x[1])
    label, conf = top[0], top[1]
    color = DISEASE_COLORS.get(label, "#F59E0B")
    display = label.replace("_", " ").title()
    remedy = REMEDIES.get(label, "No remedy found.")

    st.markdown(
        f'<div class="badge" style="background:{color}22;color:{color};'
        f'border:1px solid {color};">⚠ {display.upper()} — {conf:.0%}</div>',
        unsafe_allow_html=True)

    steps_html = ""
    for i, sentence in enumerate(remedy.split(". ")):
        sentence = sentence.strip()
        if not sentence:
            continue
        if not sentence.endswith("."):
            sentence += "."
        steps_html += (
            f'<div class="remedy-step"><span class="step-num">{i+1:02d}</span>{sentence}</div>'
        )

    st.markdown(
        f'<div class="remedy-box">'
        f'<div style="color:{color};font-size:16px;font-weight:bold;">{display}</div>'
        f'<div style="color:#64748B;font-size:11px;margin:8px 0;">RECOMMENDED TREATMENT</div>'
        f'{steps_html}</div>',
        unsafe_allow_html=True)

    if len(dets) > 1:
        with st.expander(f"All {len(dets)} detections"):
            for (lbl, c, *_ ) in sorted(dets, key=lambda x: -x[1]):
                st.write(f"- **{lbl.replace('_',' ').title()}** — {c:.0%}")

    return label


def maybe_speak(label: str):
    """Optional in-browser voice alert using gTTS (server has no speakers)."""
    if not label:
        return
    last = st.session_state.get("last_spoken")
    if last == label:
        return
    st.session_state["last_spoken"] = label
    try:
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(text=label.replace("_", " "), lang="en").write_to_fp(buf)
        buf.seek(0)
        st.audio(buf, format="audio/mp3", autoplay=True)
    except Exception:
        # No internet access to Google TTS, or gTTS not installed — skip silently.
        pass


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ Settings")
    conf_threshold = st.slider("Detection confidence", 0.10, 0.95, 0.45, 0.05)
    voice_alerts = st.checkbox("🔊 Voice alerts (needs internet)", value=False)
    st.markdown("---")
    st.markdown("### 🕹 About RC car control")
    st.caption(
        "The original desktop app could drive a 4WD Arduino RC car over a "
        "COM/serial port. A hosted web app has no access to your local "
        "hardware, so that control panel isn't included here. Run the "
        "original Tkinter script locally if you need Arduino control."
    )

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown(
    "<h2 style='color:#34D399;'>🌾 AgriBot — Rice Disease Detector</h2>"
    "<p style='color:#475569;'>YOLO-based rice disease detection with a built-in remedy advisor</p>",
    unsafe_allow_html=True,
)

try:
    model = load_model()
    model_ok = True
except Exception as ex:
    model_ok = False
    st.error(
        f"Could not load `{MODEL_PATH}`. Make sure the model file is in the "
        f"repo root and `ultralytics` is installed.\n\n**Details:** {ex}"
    )

# ─────────────────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────────────────
col_input, col_result, col_remedy = st.columns([1, 1.2, 1])

with col_input:
    st.markdown("#### 📷 Input")
    mode = st.radio("Source", ["Upload image", "Camera snapshot"], label_visibility="collapsed")
    image_bgr = None

    if mode == "Upload image":
        upload = st.file_uploader("Upload a rice plant image", type=["jpg", "jpeg", "png", "bmp"])
        if upload is not None:
            pil_img = Image.open(upload).convert("RGB")
            image_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    else:
        snap = st.camera_input("Point at a rice plant and take a photo")
        if snap is not None:
            pil_img = Image.open(snap).convert("RGB")
            image_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

with col_result:
    st.markdown("#### 🖼 Detection result")
    top_label = None
    if image_bgr is not None and model_ok:
        with st.spinner("Running detection…"):
            t0 = time.time()
            annotated, dets = run_detection(image_bgr, model, conf_threshold)
            elapsed = time.time() - t0
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, use_container_width=True)
        st.caption(f"Inference time: {elapsed:.2f}s · {len(dets)} detection(s)")
    else:
        st.info("Upload an image or take a camera snapshot to scan for disease.")
        dets = []

with col_remedy:
    st.markdown("#### 💊 Remedy advisor")
    top_label = render_remedy_panel(dets)
    if voice_alerts:
        maybe_speak(top_label)

st.markdown("---")
st.caption(
    "Classes: " + ", ".join(c.replace("_", " ").title() for c in CLASS_NAMES)
)
