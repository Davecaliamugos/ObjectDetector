# ============================================================
# PHASE 1: IMPORTS
# ============================================================
import streamlit as st
import cv2
import numpy as np
import time
import collections
from ultralytics import YOLO
from PIL import Image
import torch

cv2.ocl.setUseOpenCL(False)


# ============================================================
# UNIVERSAL FRAME SANITISER
# ============================================================
def to_cpu_mat(frame) -> np.ndarray:
    if isinstance(frame, cv2.UMat):
        frame = frame.get()
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)
    if frame.dtype != np.uint8:
        if frame.dtype in (np.float32, np.float64):
            frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 1:
        frame = cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if not frame.flags["C_CONTIGUOUS"]:
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
    return frame


# ============================================================
# PHASE 2: PAGE CONFIG & PROFESSIONAL CSS
# ============================================================
st.set_page_config(
    page_title="Vision AI — Object Detection",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<link href="https://cdn.jsdelivr.net/npm/remixicon@4.1.0/fonts/remixicon.css" rel="stylesheet">

<style>
    /* ── GLOBAL RESET ─────────────────────────────────── */
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-tertiary: #1a1a26;
        --bg-card: #16161f;
        --bg-elevated: #1e1e2a;
        --border-subtle: rgba(255, 255, 255, 0.06);
        --border-default: rgba(255, 255, 255, 0.08);
        --border-strong: rgba(255, 255, 255, 0.12);
        --text-primary: #f0f0f5;
        --text-secondary: #9898a6;
        --text-tertiary: #6a6a7a;
        --text-muted: #4a4a58;
        --accent-green: #00d47e;
        --accent-green-dim: rgba(0, 212, 126, 0.12);
        --accent-green-glow: rgba(0, 212, 126, 0.25);
        --accent-blue: #3b82f6;
        --accent-blue-dim: rgba(59, 130, 246, 0.12);
        --accent-red: #ef4444;
        --accent-red-dim: rgba(239, 68, 68, 0.12);
        --accent-amber: #f59e0b;
        --accent-amber-dim: rgba(245, 158, 11, 0.12);
        --accent-violet: #8b5cf6;
        --accent-violet-dim: rgba(139, 92, 246, 0.12);
        --radius-sm: 6px;
        --radius-md: 10px;
        --radius-lg: 14px;
        --radius-xl: 20px;
        --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
        --shadow-md: 0 4px 12px rgba(0,0,0,0.4);
        --shadow-lg: 0 8px 32px rgba(0,0,0,0.5);
        --shadow-glow-green: 0 0 20px rgba(0, 212, 126, 0.15);
        --shadow-glow-red: 0 0 20px rgba(239, 68, 68, 0.15);
        --transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }

    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }
    code, .mono { font-family: 'JetBrains Mono', monospace !important; }

    .stApp {
        background: var(--bg-primary) !important;
    }

    /* ── SCROLLBAR ────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb {
        background: var(--bg-elevated);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover { background: #2a2a3a; }

    /* ── HEADER ───────────────────────────────────────── */
    .vision-header {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 16px;
        padding: 32px 20px 8px;
    }
    .vision-logo {
        width: 44px; height: 44px;
        background: linear-gradient(135deg, var(--accent-green), var(--accent-blue));
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: var(--shadow-glow-green);
    }
    .vision-logo i {
        font-size: 22px;
        color: white;
    }
    .vision-title {
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: var(--text-primary);
        line-height: 1.1;
    }
    .vision-title span {
        color: var(--accent-green);
    }
    .vision-subtitle {
        text-align: center;
        color: var(--text-tertiary);
        font-size: 0.85rem;
        font-weight: 400;
        letter-spacing: 0.02em;
        margin-bottom: 24px;
    }

    /* ── DIVIDER ──────────────────────────────────────── */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-default), transparent);
        margin: 16px 0;
    }

    /* ── METRIC CARDS (top bar) ────────────────────────── */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 20px;
    }
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: 16px 18px;
        transition: var(--transition);
    }
    .metric-card:hover {
        border-color: var(--border-strong);
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
    }
    .metric-label {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 0.7rem;
        font-weight: 500;
        color: var(--text-tertiary);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 8px;
    }
    .metric-label i {
        font-size: 12px;
        opacity: 0.7;
    }
    .metric-value {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
        font-family: 'JetBrains Mono', monospace !important;
    }
    .metric-value.green { color: var(--accent-green); }
    .metric-value.blue  { color: var(--accent-blue); }
    .metric-value.amber { color: var(--accent-amber); }

    /* ── STAT PANEL ───────────────────────────────────── */
    .stat-panel {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: 18px;
        margin-bottom: 12px;
        transition: var(--transition);
    }
    .stat-panel:hover {
        border-color: var(--border-default);
    }
    .stat-panel-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 14px;
    }
    .stat-panel-icon {
        width: 28px; height: 28px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 13px;
    }
    .stat-panel-icon.green {
        background: var(--accent-green-dim);
        color: var(--accent-green);
    }
    .stat-panel-icon.red {
        background: var(--accent-red-dim);
        color: var(--accent-red);
    }
    .stat-panel-icon.blue {
        background: var(--accent-blue-dim);
        color: var(--accent-blue);
    }
    .stat-panel-icon.violet {
        background: var(--accent-violet-dim);
        color: var(--accent-violet);
    }
    .stat-panel-icon.amber {
        background: var(--accent-amber-dim);
        color: var(--accent-amber);
    }
    .stat-panel-title {
        font-size: 0.72rem;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* ── FPS DISPLAY ──────────────────────────────────── */
    .fps-display {
        display: flex;
        align-items: baseline;
        gap: 4px;
        margin-bottom: 4px;
    }
    .fps-value {
        font-size: 2rem;
        font-weight: 800;
        color: var(--accent-green);
        font-family: 'JetBrains Mono', monospace !important;
        letter-spacing: -0.03em;
        line-height: 1;
    }
    .fps-unit {
        font-size: 0.75rem;
        font-weight: 500;
        color: var(--text-tertiary);
        text-transform: uppercase;
    }
    .fps-meta {
        font-size: 0.7rem;
        color: var(--text-muted);
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ── ALERT STATES ─────────────────────────────────── */
    .alert-human {
        background: linear-gradient(135deg,
            rgba(239, 68, 68, 0.15),
            rgba(239, 68, 68, 0.05));
        border: 1px solid rgba(239, 68, 68, 0.25);
        border-radius: var(--radius-md);
        padding: 12px 16px;
        display: flex;
        align-items: center;
        gap: 10px;
        box-shadow: var(--shadow-glow-red);
        animation: pulse-red 2s ease-in-out infinite;
    }
    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 0 15px rgba(239, 68, 68, 0.1); }
        50%      { box-shadow: 0 0 25px rgba(239, 68, 68, 0.2); }
    }
    .alert-human .alert-dot {
        width: 8px; height: 8px;
        background: var(--accent-red);
        border-radius: 50%;
        animation: blink 1s ease-in-out infinite;
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50%      { opacity: 0.3; }
    }
    .alert-human .alert-text {
        font-size: 0.78rem;
        font-weight: 600;
        color: var(--accent-red);
        letter-spacing: 0.03em;
    }
    .alert-clear {
        background: var(--accent-green-dim);
        border: 1px solid rgba(0, 212, 126, 0.15);
        border-radius: var(--radius-md);
        padding: 12px 16px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .alert-clear .alert-dot {
        width: 8px; height: 8px;
        background: var(--accent-green);
        border-radius: 50%;
    }
    .alert-clear .alert-text {
        font-size: 0.78rem;
        font-weight: 500;
        color: var(--accent-green);
    }

    /* ── DETECTION LIST ───────────────────────────────── */
    .det-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid var(--border-subtle);
    }
    .det-item:last-child { border-bottom: none; }
    .det-item-left {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .det-dot {
        width: 6px; height: 6px;
        border-radius: 50%;
    }
    .det-dot.person { background: var(--accent-red); }
    .det-dot.object { background: var(--accent-blue); }
    .det-name {
        font-size: 0.8rem;
        font-weight: 500;
        color: var(--text-primary);
        text-transform: capitalize;
    }
    .det-count {
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--text-secondary);
        background: var(--bg-elevated);
        padding: 2px 10px;
        border-radius: 20px;
        font-family: 'JetBrains Mono', monospace !important;
    }
    .det-empty {
        text-align: center;
        padding: 20px 0;
        color: var(--text-muted);
        font-size: 0.8rem;
    }

    /* ── SESSION STATS ────────────────────────────────── */
    .session-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 0;
    }
    .session-label {
        font-size: 0.75rem;
        color: var(--text-tertiary);
    }
    .session-value {
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--text-primary);
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ── VIDEO CONTAINER ──────────────────────────────── */
    .video-container {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-xl);
        padding: 8px;
        position: relative;
        overflow: hidden;
    }
    .video-container img {
        border-radius: calc(var(--radius-xl) - 6px);
    }
    .video-badge {
        position: absolute;
        top: 18px;
        left: 18px;
        display: flex;
        align-items: center;
        gap: 6px;
        background: rgba(0, 0, 0, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 5px 12px;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .video-badge .live-dot {
        width: 7px; height: 7px;
        background: var(--accent-red);
        border-radius: 50%;
        animation: blink 1.2s ease-in-out infinite;
    }
    .video-badge span {
        font-size: 0.68rem;
        font-weight: 600;
        color: var(--text-primary);
        letter-spacing: 0.04em;
    }

    /* ── HERO / IDLE STATE ────────────────────────────── */
    .hero-section {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-xl);
        padding: 60px 30px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .hero-section::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(
            ellipse at center,
            rgba(0, 212, 126, 0.03) 0%,
            transparent 60%
        );
        pointer-events: none;
    }
    .hero-icon {
        width: 72px; height: 72px;
        margin: 0 auto 24px;
        background: linear-gradient(135deg,
            var(--accent-green-dim),
            var(--accent-blue-dim));
        border-radius: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 1px solid var(--border-default);
    }
    .hero-icon i {
        font-size: 30px;
        background: linear-gradient(135deg, var(--accent-green), var(--accent-blue));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 8px;
        letter-spacing: -0.02em;
    }
    .hero-desc {
        font-size: 0.88rem;
        color: var(--text-tertiary);
        max-width: 500px;
        margin: 0 auto 30px;
        line-height: 1.6;
    }
    .hero-features {
        display: flex;
        justify-content: center;
        gap: 10px;
        flex-wrap: wrap;
    }
    .hero-chip {
        display: flex;
        align-items: center;
        gap: 6px;
        background: var(--bg-elevated);
        border: 1px solid var(--border-subtle);
        padding: 7px 16px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
        color: var(--text-secondary);
        transition: var(--transition);
    }
    .hero-chip:hover {
        border-color: var(--border-strong);
        transform: translateY(-1px);
    }
    .hero-chip i {
        font-size: 13px;
    }
    .hero-chip.green i { color: var(--accent-green); }
    .hero-chip.blue i  { color: var(--accent-blue); }
    .hero-chip.red i   { color: var(--accent-red); }
    .hero-chip.amber i { color: var(--accent-amber); }

    /* ── CLASSES EXPANDER ─────────────────────────────── */
    .class-tag {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: var(--bg-elevated);
        border: 1px solid var(--border-subtle);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.72rem;
        color: var(--text-secondary);
        font-family: 'JetBrains Mono', monospace !important;
        margin: 2px;
    }
    .class-tag.person {
        border-color: rgba(239, 68, 68, 0.2);
        color: var(--accent-red);
    }

    /* ── BUTTONS ──────────────────────────────────────── */
    .stButton > button {
        width: 100%;
        border-radius: var(--radius-md) !important;
        height: 48px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.01em !important;
        border: none !important;
        transition: var(--transition) !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--accent-green), #00b368) !important;
        color: #000 !important;
        box-shadow: var(--shadow-glow-green) !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 0 30px rgba(0, 212, 126, 0.25) !important;
    }

    /* ── SIDEBAR ──────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 24px;
    }
    .sidebar-section {
        margin-bottom: 20px;
    }
    .sidebar-section-title {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--text-tertiary);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--border-subtle);
    }
    .sidebar-section-title i {
        font-size: 13px;
        opacity: 0.6;
    }
    .sys-info-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 0;
        border-bottom: 1px solid var(--border-subtle);
    }
    .sys-info-row:last-child { border-bottom: none; }
    .sys-info-label {
        font-size: 0.73rem;
        color: var(--text-tertiary);
    }
    .sys-info-value {
        font-size: 0.73rem;
        font-weight: 600;
        color: var(--text-primary);
        font-family: 'JetBrains Mono', monospace !important;
    }
    .sys-badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.68rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace !important;
    }
    .sys-badge.gpu {
        background: var(--accent-green-dim);
        color: var(--accent-green);
    }
    .sys-badge.cpu {
        background: var(--accent-amber-dim);
        color: var(--accent-amber);
    }

    /* ── HIDE STREAMLIT DEFAULTS ──────────────────────── */
    #MainMenu, header, footer,
    div[data-testid="stToolbar"],
    div[data-testid="stDecoration"],
    div[data-testid="stStatusWidget"] {
        display: none !important;
    }
    div[data-testid="metric-container"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# PHASE 3: MODEL LOADER
# ============================================================
@st.cache_resource
def load_yolo_model(model_size: str = "yolov8n"):
    try:
        model = YOLO(f"{model_size}.pt")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model._infer_device = device
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


# ============================================================
# PHASE 4: COLOUR PALETTE
# ============================================================
_RNG = np.random.default_rng(42)
_CLASS_COLORS = _RNG.integers(80, 220, size=(300, 3), dtype=np.uint8)

_PERSON_COLOR = (71, 68, 239)  # BGR for #EF4447
_SPECIAL_COLORS = {
    "car":        (246, 130,  59),
    "truck":      (200, 100,  50),
    "motorcycle": (11, 158, 245),
    "bicycle":    (126, 212,   0),
    "dog":        (126, 232, 100),
    "cat":        (200, 232, 100),
    "bird":       (255, 200, 100),
    "cell phone": (246,  92, 139),
    "laptop":     (246, 92,  200),
}


def _get_color(class_name: str, class_id: int) -> tuple:
    if class_name.lower() == "person":
        return _PERSON_COLOR
    if class_name.lower() in _SPECIAL_COLORS:
        return _SPECIAL_COLORS[class_name.lower()]
    c = _CLASS_COLORS[class_id % len(_CLASS_COLORS)]
    return int(c[0]), int(c[1]), int(c[2])


# ============================================================
# PHASE 5: DETECTION ENGINE
# ============================================================
def detect_objects(frame, model, confidence_threshold=0.5, iou_threshold=0.45):
    frame = to_cpu_mat(frame)
    yolo_input = frame.copy()
    detections = []
    detection_counts = collections.Counter()
    human_detected = False

    results = model(
        yolo_input, conf=confidence_threshold,
        iou=iou_threshold, verbose=False, stream=False,
    )

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            detections.append({
                "class": class_name, "confidence": confidence,
                "bbox": (x1, y1, x2, y2), "class_id": class_id,
            })
            detection_counts[class_name] += 1
            if class_name.lower() == "person":
                human_detected = True

    canvas = to_cpu_mat(frame.copy())
    annotated = draw_detections(canvas, detections, human_detected)
    annotated = to_cpu_mat(annotated)
    return annotated, detections, detection_counts, human_detected


def draw_detections(frame, detections, human_detected=False):
    height, width = frame.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        class_name = det["class"]
        confidence = det["confidence"]
        class_id = det["class_id"]
        color = _get_color(class_name, class_id)
        is_person = class_name.lower() == "person"
        thickness = 2

        # Draw rounded-look rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Refined corner markers
        clen, cthk = 15, 3
        for px, py, sx, sy in [
            (x1, y1, 1, 1), (x2, y1, -1, 1),
            (x1, y2, 1, -1), (x2, y2, -1, -1),
        ]:
            cv2.line(frame, (px, py), (px + sx * clen, py), color, cthk)
            cv2.line(frame, (px, py), (px, py + sy * clen), color, cthk)

        # Clean label
        label = f"{class_name}  {confidence:.0%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        fscl = 0.48
        fthk = 1
        (tw, th), baseline = cv2.getTextSize(label, font, fscl, fthk)
        label_y = max(y1 - 6, th + 8)

        # Label background with padding
        pad_x, pad_y = 6, 4
        cv2.rectangle(
            frame,
            (x1, label_y - th - pad_y * 2),
            (x1 + tw + pad_x * 2, label_y + 2),
            color, -1,
        )
        cv2.putText(
            frame, label,
            (x1 + pad_x, label_y - pad_y),
            font, fscl, (255, 255, 255), fthk, cv2.LINE_AA,
        )

        # Subtle centre crosshair
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA)

        # Distance hint for persons
        if is_person:
            bh = y2 - y1
            if bh > 10:
                dist_text = f"~{max(1, int(500 / bh))}m"
                cv2.putText(
                    frame, dist_text, (x1, y2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (180, 200, 220), 1, cv2.LINE_AA,
                )

    # Bottom info bar
    bar_h = 36
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, height - bar_h), (width, height), (10, 10, 15), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Object count (left)
    cv2.putText(
        frame,
        f"{len(detections)} object{'s' if len(detections) != 1 else ''} detected",
        (12, height - 11),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        (160, 170, 180), 1, cv2.LINE_AA,
    )

    # Alert text (right)
    if human_detected:
        alert = "HUMAN DETECTED"
        (aw, _), _ = cv2.getTextSize(alert, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(
            frame, alert,
            (width - aw - 12, height - 11),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (71, 68, 239), 1, cv2.LINE_AA,
        )

    return frame


# ============================================================
# PHASE 6: PIL CONVERSION
# ============================================================
def frame_to_pil(frame):
    frame = to_cpu_mat(frame)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    return Image.fromarray(rgb)


# ============================================================
# PHASE 7: FPS COUNTER
# ============================================================
class FPSCounter:
    def __init__(self, buffer_size=30):
        self._ts = collections.deque(maxlen=buffer_size)

    def update(self):
        self._ts.append(time.perf_counter())

    def get_fps(self):
        if len(self._ts) < 2:
            return 0.0
        elapsed = self._ts[-1] - self._ts[0]
        return 0.0 if elapsed == 0 else (len(self._ts) - 1) / elapsed


# ============================================================
# PHASE 8: SIDEBAR
# ============================================================
def render_sidebar():
    # Model section
    st.sidebar.markdown("""
    <div class="sidebar-section-title">
        <i class="ri-cpu-line"></i> Model Configuration
    </div>""", unsafe_allow_html=True)

    model_options = {
        "YOLOv8 Nano — Fastest":        "yolov8n",
        "YOLOv8 Small — Balanced":       "yolov8s",
        "YOLOv8 Medium — Accurate":      "yolov8m",
        "YOLOv8 Large — High Accuracy":  "yolov8l",
        "YOLOv8 XLarge — Maximum":       "yolov8x",
    }
    label = st.sidebar.selectbox(
        "Model variant",
        list(model_options.keys()),
        index=0,
        label_visibility="collapsed",
    )

    # Thresholds
    st.sidebar.markdown("""
    <div class="sidebar-section-title" style="margin-top: 16px;">
        <i class="ri-equalizer-line"></i> Detection Thresholds
    </div>""", unsafe_allow_html=True)

    confidence = st.sidebar.slider(
        "Confidence", 0.10, 1.0, 0.50, 0.05,
        help="Minimum confidence score to accept a detection",
    )
    iou = st.sidebar.slider(
        "NMS IoU", 0.10, 1.0, 0.45, 0.05,
        help="Intersection-over-Union threshold for non-max suppression",
    )

    # Camera
    st.sidebar.markdown("""
    <div class="sidebar-section-title" style="margin-top: 16px;">
        <i class="ri-camera-line"></i> Camera Settings
    </div>""", unsafe_allow_html=True)

    camera_index = st.sidebar.number_input(
        "Camera Index", min_value=0, max_value=10, value=0,
        help="Device index — 0 is usually the default webcam",
    )
    res_options = {
        "640 × 480": (640, 480),
        "800 × 600": (800, 600),
        "1280 × 720  HD": (1280, 720),
    }
    res_label = st.sidebar.selectbox("Resolution", list(res_options.keys()), index=0)
    resolution = res_options[res_label]

    # Display
    st.sidebar.markdown("""
    <div class="sidebar-section-title" style="margin-top: 16px;">
        <i class="ri-settings-3-line"></i> Display Options
    </div>""", unsafe_allow_html=True)

    show_fps = st.sidebar.checkbox("Show FPS overlay", value=True)
    flip = st.sidebar.checkbox("Mirror camera", value=True)
    max_fps = st.sidebar.slider("Frame rate limit", 5, 60, 30)

    # System info
    st.sidebar.markdown("""
    <div class="sidebar-section-title" style="margin-top: 16px;">
        <i class="ri-information-line"></i> System
    </div>""", unsafe_allow_html=True)

    is_gpu = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0)[:24] if is_gpu else "—"
    badge = "gpu" if is_gpu else "cpu"
    badge_text = "CUDA" if is_gpu else "CPU"

    st.sidebar.markdown(f"""
    <div style="background: var(--bg-tertiary); border-radius: var(--radius-md);
                padding: 12px; border: 1px solid var(--border-subtle);">
        <div class="sys-info-row">
            <span class="sys-info-label">Compute</span>
            <span class="sys-badge {badge}">{badge_text}</span>
        </div>
        <div class="sys-info-row">
            <span class="sys-info-label">GPU</span>
            <span class="sys-info-value">{gpu_name}</span>
        </div>
        <div class="sys-info-row">
            <span class="sys-info-label">PyTorch</span>
            <span class="sys-info-value">{torch.__version__}</span>
        </div>
        <div class="sys-info-row">
            <span class="sys-info-label">OpenCL</span>
            <span class="sys-info-value">{"on" if cv2.ocl.useOpenCL() else "off"}</span>
        </div>
    </div>""", unsafe_allow_html=True)

    return {
        "model": model_options[label],
        "confidence": confidence,
        "iou": iou,
        "camera_index": int(camera_index),
        "resolution": resolution,
        "show_fps": show_fps,
        "flip_camera": flip,
        "max_fps": max_fps,
    }


# ============================================================
# PHASE 9: st.image COMPAT
# ============================================================
def _st_image(placeholder, pil_img, caption=""):
    if "_img_kw" not in st.session_state:
        import inspect
        params = inspect.signature(st.image).parameters
        st.session_state["_img_kw"] = (
            "use_container_width" if "use_container_width" in params else "width"
        )
    kw = st.session_state["_img_kw"]
    if kw == "use_container_width":
        placeholder.image(pil_img, use_container_width=True, caption=caption)
    else:
        placeholder.image(pil_img, width=700, caption=caption)


# ============================================================
# PHASE 10: DETECTION LOOP
# ============================================================
def run_detection(config, model):
    # Layout
    col_main, col_stats = st.columns([3, 1], gap="medium")

    with col_main:
        video_ph = st.empty()

    with col_stats:
        fps_ph = st.empty()
        alert_ph = st.empty()
        detlist_ph = st.empty()
        session_ph = st.empty()

    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        if st.button("Stop Detection", type="primary", key="stop_btn"):
            st.session_state["running"] = False

    # Camera
    cap = cv2.VideoCapture(config["camera_index"], cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(config["camera_index"])
    if not cap.isOpened():
        st.error(f"Cannot open camera {config['camera_index']}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["resolution"][0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["resolution"][1])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, config["max_fps"])

    fps_counter = FPSCounter(30)
    frame_count = 0
    total_detects = 0
    start_time = time.perf_counter()
    min_frame_time = 1.0 / config["max_fps"]
    read_failures = 0
    MAX_FAILURES = 10

    try:
        while st.session_state.get("running", False):
            t0 = time.perf_counter()

            ret, raw = cap.read()
            if not ret or raw is None:
                read_failures += 1
                if read_failures >= MAX_FAILURES:
                    st.warning("Camera stopped sending frames.")
                    break
                time.sleep(0.05)
                continue
            read_failures = 0

            frame = to_cpu_mat(raw)
            if config["flip_camera"]:
                frame = to_cpu_mat(cv2.flip(frame, 1))

            try:
                annotated, detections, det_counts, human_detected = detect_objects(
                    frame, model, config["confidence"], config["iou"]
                )
            except Exception as e:
                st.warning(f"Inference error: {e}")
                annotated, detections = frame, []
                det_counts, human_detected = collections.Counter(), False

            fps_counter.update()
            frame_count += 1
            total_detects += len(detections)
            current_fps = fps_counter.get_fps()

            annotated = to_cpu_mat(annotated)

            # FPS + model overlay on frame
            if config["show_fps"]:
                cv2.putText(
                    annotated, f"FPS {current_fps:.1f}",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (126, 212, 0), 2, cv2.LINE_AA,
                )
                cv2.putText(
                    annotated, config["model"].upper(),
                    (12, 52), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (160, 170, 180), 1, cv2.LINE_AA,
                )

            # Timestamp
            ts = time.strftime("%H:%M:%S")
            h, w = annotated.shape[:2]
            (tsw, _), _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.putText(
                annotated, ts,
                (w - tsw - 10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 160), 1, cv2.LINE_AA,
            )

            # Render video
            _st_image(video_ph, frame_to_pil(annotated), "")

            # ── Stats Panel: FPS ────────────────────────
            fps_ph.markdown(f"""
            <div class="stat-panel">
                <div class="stat-panel-header">
                    <div class="stat-panel-icon green">
                        <i class="ri-speed-line"></i>
                    </div>
                    <span class="stat-panel-title">Performance</span>
                </div>
                <div class="fps-display">
                    <span class="fps-value">{current_fps:.1f}</span>
                    <span class="fps-unit">fps</span>
                </div>
                <div class="fps-meta">Frame #{frame_count:,}</div>
            </div>""", unsafe_allow_html=True)

            # ── Stats Panel: Alert ──────────────────────
            if human_detected:
                alert_ph.markdown("""
                <div class="alert-human">
                    <div class="alert-dot"></div>
                    <span class="alert-text">Human Detected</span>
                </div>""", unsafe_allow_html=True)
            else:
                alert_ph.markdown("""
                <div class="alert-clear">
                    <div class="alert-dot"></div>
                    <span class="alert-text">Scene Clear</span>
                </div>""", unsafe_allow_html=True)

            # ── Stats Panel: Detections ─────────────────
            if det_counts:
                rows = ""
                for k, v in det_counts.most_common(8):
                    dot_class = "person" if k == "person" else "object"
                    rows += f"""
                    <div class="det-item">
                        <div class="det-item-left">
                            <div class="det-dot {dot_class}"></div>
                            <span class="det-name">{k}</span>
                        </div>
                        <span class="det-count">{v}</span>
                    </div>"""
                detlist_ph.markdown(f"""
                <div class="stat-panel">
                    <div class="stat-panel-header">
                        <div class="stat-panel-icon blue">
                            <i class="ri-focus-3-line"></i>
                        </div>
                        <span class="stat-panel-title">Detected Objects</span>
                    </div>
                    {rows}
                </div>""", unsafe_allow_html=True)
            else:
                detlist_ph.markdown("""
                <div class="stat-panel">
                    <div class="stat-panel-header">
                        <div class="stat-panel-icon blue">
                            <i class="ri-focus-3-line"></i>
                        </div>
                        <span class="stat-panel-title">Detected Objects</span>
                    </div>
                    <div class="det-empty">
                        <i class="ri-scan-line" style="font-size:20px;display:block;margin-bottom:6px;"></i>
                        Scanning environment
                    </div>
                </div>""", unsafe_allow_html=True)

            # ── Stats Panel: Session ────────────────────
            elapsed = time.perf_counter() - start_time
            m, s = divmod(int(elapsed), 60)
            session_ph.markdown(f"""
            <div class="stat-panel">
                <div class="stat-panel-header">
                    <div class="stat-panel-icon violet">
                        <i class="ri-bar-chart-2-line"></i>
                    </div>
                    <span class="stat-panel-title">Session</span>
                </div>
                <div class="session-row">
                    <span class="session-label">Total detections</span>
                    <span class="session-value">{total_detects:,}</span>
                </div>
                <div class="session-row">
                    <span class="session-label">Runtime</span>
                    <span class="session-value">{m:02d}:{s:02d}</span>
                </div>
                <div class="session-row">
                    <span class="session-label">Unique classes</span>
                    <span class="session-value">{len(det_counts)}</span>
                </div>
                <div class="session-row">
                    <span class="session-label">Avg det/frame</span>
                    <span class="session-value">{(total_detects / max(frame_count,1)):.1f}</span>
                </div>
            </div>""", unsafe_allow_html=True)

            # Frame rate limiter
            sleep_t = max(0.0, min_frame_time - (time.perf_counter() - t0))
            if sleep_t > 0:
                time.sleep(sleep_t)

    except Exception as e:
        st.error(f"Detection error: {e}")
        raise
    finally:
        cap.release()
        st.session_state["running"] = False
        st.info("Camera released.")


# ============================================================
# PHASE 11: MAIN
# ============================================================
def main():
    # Header
    st.markdown("""
    <div class="vision-header">
        <div class="vision-logo">
            <i class="ri-eye-line"></i>
        </div>
        <div class="vision-title">Vision<span>AI</span></div>
    </div>
    <div class="vision-subtitle">
        Real-time object detection powered by YOLOv8
    </div>
    <div class="divider"></div>
    """, unsafe_allow_html=True)

    config = render_sidebar()

    # Load model
    loader_ph = st.empty()
    with loader_ph.container():
        with st.spinner(f"Loading {config['model']}..."):
            model = load_yolo_model(config["model"])

    if model is None:
        st.error("Failed to load YOLO model.")
        st.code("pip install ultralytics torch torchvision", language="bash")
        return

    loader_ph.empty()
    device = getattr(model, "_infer_device", "cpu")

    # Metrics bar
    device_display = "CUDA" if device == "cuda" else "CPU"
    device_class = "green" if device == "cuda" else "amber"

    st.markdown(f"""
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">
                <i class="ri-cpu-line"></i> Model
            </div>
            <div class="metric-value">{config['model'].upper()}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">
                <i class="ri-flashlight-line"></i> Compute
            </div>
            <div class="metric-value {device_class}">{device_display}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">
                <i class="ri-price-tag-3-line"></i> Classes
            </div>
            <div class="metric-value blue">{len(model.names)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">
                <i class="ri-camera-line"></i> Camera
            </div>
            <div class="metric-value">ID {config['camera_index']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Classes expander
    with st.expander("View all detectable classes"):
        class_tags = ""
        for cls in model.names.values():
            tag_class = "person" if cls == "person" else ""
            class_tags += f'<span class="class-tag {tag_class}">{cls}</span>'
        st.markdown(f"""
        <div style="line-height: 2.2;">{class_tags}</div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # State
    if "running" not in st.session_state:
        st.session_state["running"] = False

    # Start button
    _, col_btn, _ = st.columns([1, 2, 1])
    with col_btn:
        if not st.session_state["running"]:
            if st.button("Start Live Detection", type="primary", key="start_btn"):
                st.session_state["running"] = True
                st.rerun()
        else:
            st.markdown("""
            <div style="display:flex;align-items:center;justify-content:center;
                        gap:8px;padding:10px;background:var(--accent-green-dim);
                        border:1px solid rgba(0,212,126,0.15);border-radius:var(--radius-md);">
                <div style="width:8px;height:8px;background:var(--accent-green);
                            border-radius:50%;animation:blink 1.2s ease-in-out infinite;"></div>
                <span style="font-size:0.82rem;font-weight:500;color:var(--accent-green);">
                    Detection active
                </span>
            </div>""", unsafe_allow_html=True)

    if st.session_state["running"]:
        run_detection(config, model)
    else:
        st.markdown("""
        <div class="hero-section">
            <div class="hero-icon">
                <i class="ri-camera-lens-line"></i>
            </div>
            <div class="hero-title">Ready to Detect</div>
            <div class="hero-desc">
                Configure your model and camera settings in the sidebar,
                then start live detection to identify objects in real-time
                using state-of-the-art computer vision.
            </div>
            <div class="hero-features">
                <div class="hero-chip green">
                    <i class="ri-speed-line"></i>
                    Real-time inference
                </div>
                <div class="hero-chip blue">
                    <i class="ri-focus-3-line"></i>
                    80+ object classes
                </div>
                <div class="hero-chip red">
                    <i class="ri-user-search-line"></i>
                    Human detection
                </div>
                <div class="hero-chip amber">
                    <i class="ri-line-chart-line"></i>
                    Live analytics
                </div>
            </div>
        </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()