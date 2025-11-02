# streamlit_app.py  — Lite & stable (Mac friendly)

import os, json, math
from collections import deque

# Reduce thread contention on Mac
os.environ.setdefault("OMP_NUM_THREADS", "1")

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import tensorflow as tf
from streamlit.errors import StreamlitAPIException

# ---- must be first Streamlit command ----
try:
    st.set_page_config(page_title="Emotion-Aware AI Companion", layout="wide")
except StreamlitAPIException:
    pass

MODEL_PATH = "saved_models/fer_mobilenetv2.keras"
CLASS_PATH = "saved_models/class_names.json"
IMG_SIZE   = (160, 160)

DEFAULT_SMOOTH = 3
DEFAULT_MARGIN = 0.15   # small padding
SHOW_BARS      = True

# ---------- Load model/classes once ----------
@st.cache_resource
def load_model_and_classes():
    m = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_PATH) as f:
        classes = json.load(f)
    return m, classes

model, class_names = load_model_and_classes()
num_classes = len(class_names)

# ---------- Fast detector ----------
mp_fd = mp.solutions.face_detection
fd = mp_fd.FaceDetection(min_detection_confidence=0.60)

def square_crop_with_margin(frame, x, y, w, h, margin):
    H, W = frame.shape[:2]
    size = int(max(w, h) * (1.0 + margin))
    cx, cy = x + w // 2, y + h // 2
    x1 = max(0, cx - size // 2); y1 = max(0, cy - size // 2)
    x2 = min(W, x1 + size);      y2 = min(H, y1 + size)
    x1 = max(0, x2 - size);      y1 = max(0, y2 - size)
    return frame[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)

def preprocess_face(bgr_square):
    gray = cv2.cvtColor(bgr_square, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, IMG_SIZE, interpolation=cv2.INTER_AREA)
    face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
    face = face.astype("float32")   # keep 0..255 (MobileNetV2 preprocess is inside model)
    return np.expand_dims(face, 0)

def draw_bars(frame, probs, names, x0=10, y0=30, width=220, height=18, gap=6):
    for i, (p, name) in enumerate(zip(probs, names)):
        y = y0 + i*(height+gap)
        cv2.rectangle(frame, (x0, y), (x0+width, y+height), (40,40,40), -1)
        cv2.rectangle(frame, (x0, y), (x0+int(width*float(p)), y+height), (0,255,0), -1)
        cv2.putText(frame, f"{name[:10]:10s} {p:.2f}", (x0+width+8, y+height-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

# ---------- UI ----------
st.title("😊 Emotion-Aware AI Companion (Lite)")

with st.sidebar:
    st.header("Settings")
    smooth_n = st.slider("Smoothing (frames)", 1, 7, DEFAULT_SMOOTH, step=2)
    margin   = st.slider("Face padding", 0.10, 0.25, DEFAULT_MARGIN, step=0.01)
    show_bars = st.checkbox("Show confidence bars", value=SHOW_BARS)
    st.caption("If video stutters: close other camera apps; reduce browser zoom; keep room well-lit.")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

state = {"hist": deque(maxlen=max(1, smooth_n))}

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")

    # downscale to 640x480 to keep it smooth
    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)

    H, W = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = fd.process(rgb)

    probs = np.zeros(num_classes, dtype=np.float32)

    if res.detections:
        # take best detection
        det = max(res.detections, key=lambda d: d.score[0] if d.score else 0.0)
        bb = det.location_data.relative_bounding_box
        x = int(bb.xmin * W); y = int(bb.ymin * H)
        w = int(bb.width * W); h = int(bb.height * H)

        crop, (x1, y1, sw, sh) = square_crop_with_margin(img, x, y, w, h, margin=margin)
        if crop.size > 0:
            inp = preprocess_face(crop)
            # update smoothing buffer length if user changed it
            if state["hist"].maxlen != max(1, smooth_n):
                state["hist"] = deque(list(state["hist"]), maxlen=max(1, smooth_n))
            p = model.predict(inp, verbose=0)[0]
            state["hist"].append(p)
            probs = np.mean(state["hist"], axis=0) if len(state["hist"]) > 1 else p

            idx = int(np.argmax(probs))
            label = f"{class_names[idx]} ({probs[idx]:.2f})"
            cv2.rectangle(img, (x1, y1), (x1+sw, y1+sh), (0,255,0), 2)
            cv2.putText(img, label, (x1, max(20, y1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)

    if show_bars:
        draw_bars(img, probs, class_names, x0=10, y0=40)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="fer-lite",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
    video_frame_callback=video_frame_callback,
)
