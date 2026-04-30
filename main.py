
import streamlit as st
import cv2
import numpy as np
import time
import collections
import os
import threading
from ultralytics import YOLO
from PIL import Image
import torch
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    import av
    _WEBRTC_OK = True
except Exception as e:
    _WEBRTC_OK = False
    _WEBRTC_ERROR = str(e)
try:
    import mediapipe as mp
    _MP_OK = True
except Exception as e:
    mp = None
    _MP_OK = False
    _MP_ERROR = str(e)
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

cv2.ocl.setUseOpenCL(False)


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


st.set_page_config(
    page_title="Live Object Detection & Tracing",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# FIX 1: Icons — Remix Icons cannot be loaded via external <link> due to
# Streamlit's CSP. We use Unicode symbols + inline SVG paths instead.
# All ri-* classes are replaced with data-icon attributes rendered via CSS
# content or inline SVG. Below we embed a minimal icon set as a <style>
# using CSS content + a small JS snippet to swap icon placeholders.

st.markdown("""
<style>
/* ═══════════════════════════════════════════════════════════
   DESIGN TOKENS
═══════════════════════════════════════════════════════════ */
:root {
    --bg-base:        #08080e;
    --bg-surface:     #0f0f18;
    --bg-overlay:     #14141f;
    --bg-card:        #111119;
    --bg-elevated:    #181824;
    --bg-hover:       #1e1e2c;

    --border-faint:   rgba(255,255,255,0.04);
    --border-subtle:  rgba(255,255,255,0.07);
    --border-default: rgba(255,255,255,0.10);
    --border-strong:  rgba(255,255,255,0.16);

    --text-primary:   #eeeef5;
    --text-secondary: #8888a0;
    --text-tertiary:  #55556a;
    --text-muted:     #38384a;

    --green:          #00d47e;
    --green-dim:      rgba(0,212,126,0.10);
    --green-glow:     rgba(0,212,126,0.20);
    --green-border:   rgba(0,212,126,0.20);

    --blue:           #4f8ef7;
    --blue-dim:       rgba(79,142,247,0.10);

    --red:            #f05252;
    --red-dim:        rgba(240,82,82,0.10);
    --red-border:     rgba(240,82,82,0.22);
    --red-glow:       rgba(240,82,82,0.18);

    --amber:          #f0a030;
    --amber-dim:      rgba(240,160,48,0.10);

    --violet:         #9068f0;
    --violet-dim:     rgba(144,104,240,0.10);

    --cyan:           #22d3ee;
    --cyan-dim:       rgba(34,211,238,0.10);

    --r-xs:  4px;
    --r-sm:  8px;
    --r-md:  12px;
    --r-lg:  16px;
    --r-xl:  20px;
    --r-2xl: 28px;

    --ease: cubic-bezier(0.4,0,0.2,1);
    --dur:  0.18s;
}

/* ═══════════════════════════════════════════════════════════
   ICON SYSTEM — pure CSS unicode fallback (no external CDN)
   We map .icon-* classes to readable symbols so zero external
   requests are needed, fully compatible with Streamlit's CSP.
═══════════════════════════════════════════════════════════ */
.icon::before { font-style: normal; }
.icon-eye::before       { content: "◉"; }
.icon-cpu::before       { content: "⬡"; }
.icon-flash::before     { content: "⚡"; }
.icon-tag::before       { content: "⬛"; }
.icon-cam::before       { content: "◎"; }
.icon-speed::before     { content: "◈"; }
.icon-focus::before     { content: "◎"; }
.icon-chart::before     { content: "▦"; }
.icon-hand::before      { content: "✋"; }
.icon-face::before      { content: "☺"; }
.icon-spark::before     { content: "✦"; }
.icon-info::before      { content: "ℹ"; }
.icon-radar::before     { content: "◉"; }
.icon-eq::before        { content: "≡"; }
.icon-set::before       { content: "⚙"; }
.icon-lens::before      { content: "◎"; }
.icon-user::before      { content: "⬡"; }
.icon-bar::before       { content: "▦"; }

/* ═══════════════════════════════════════════════════════════
   GLOBAL BASE
═══════════════════════════════════════════════════════════ */
*, *::before, *::after {
    box-sizing: border-box;
}
html, .stApp {
    background: var(--bg-base) !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--bg-elevated); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--bg-hover); }

/* Main content padding — responsive */
.block-container {
    padding: 0 1rem 3rem !important;
    max-width: 1400px !important;
}
@media (min-width: 768px) {
    .block-container { padding: 0 2rem 3rem !important; }
}
@media (min-width: 1024px) {
    .block-container { padding: 0 2.5rem 3rem !important; }
}

/* ═══════════════════════════════════════════════════════════
   TOPBAR / HEADER — responsive
═══════════════════════════════════════════════════════════ */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px;
    padding: 16px 0 14px;
    border-bottom: 1px solid var(--border-faint);
    margin-top: 32px;
    margin-bottom: 16px;
}
@media (min-width: 768px) {
    .topbar { padding: 28px 0 20px; margin-bottom: 28px; }
}
.topbar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
}
.brand-mark {
    width: 34px; height: 34px;
    background: linear-gradient(145deg, #00d47e 0%, #4f8ef7 100%);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    font-size: 16px;
    color: #fff;
}
.brand-text { display: flex; flex-direction: column; gap: 1px; }
.brand-name {
    font-size: 1rem; font-weight: 700;
    color: var(--text-primary); letter-spacing: -0.02em; line-height: 1;
}
.brand-name span { color: var(--green); }
.brand-tagline {
    font-size: 0.65rem; font-weight: 400;
    color: var(--text-tertiary); letter-spacing: 0.03em;
}
.topbar-status {
    display: flex; align-items: center; gap: 6px;
    padding: 5px 12px;
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 99px;
    font-size: 0.70rem; font-weight: 500; color: var(--text-secondary);
}
.status-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--text-muted);
}
.status-dot.live {
    background: var(--red);
    animation: pulse-dot 1.4s ease-in-out infinite;
}
.status-dot.active { background: var(--green); }
@keyframes pulse-dot {
    0%,100% { opacity:1; transform:scale(1); }
    50% { opacity:0.5; transform:scale(0.75); }
}

/* ═══════════════════════════════════════════════════════════
   METRIC STRIP — responsive grid
═══════════════════════════════════════════════════════════ */
.metric-strip {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 8px;
    margin-bottom: 16px;
}
@media (min-width: 640px) {
    .metric-strip { grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 24px; }
}
.metric-tile {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--r-lg);
    padding: 12px 14px;
    display: flex; flex-direction: column; gap: 8px;
    transition: border-color var(--dur) var(--ease), transform var(--dur) var(--ease);
    cursor: default;
}
@media (min-width: 768px) {
    .metric-tile { padding: 18px 20px; gap: 10px; }
}
.metric-tile:hover { border-color: var(--border-default); transform: translateY(-2px); }
.metric-tile-header { display: flex; align-items: center; justify-content: space-between; }
.metric-tile-label {
    font-size: 0.62rem; font-weight: 600;
    color: var(--text-tertiary);
    text-transform: uppercase; letter-spacing: 0.08em;
}
.metric-tile-icon {
    width: 22px; height: 22px; border-radius: var(--r-sm);
    display: flex; align-items: center; justify-content: center; font-size: 11px;
}
.metric-tile-icon.green { background:var(--green-dim); color:var(--green); }
.metric-tile-icon.blue  { background:var(--blue-dim);  color:var(--blue);  }
.metric-tile-icon.amber { background:var(--amber-dim); color:var(--amber); }
.metric-tile-icon.red   { background:var(--red-dim);   color:var(--red);   }
.metric-tile-icon.violet{ background:var(--violet-dim);color:var(--violet);}
.metric-tile-value {
    font-size: 1.05rem; font-weight: 700;
    color: var(--text-primary); letter-spacing: -0.03em;
    font-family: 'Courier New', monospace; line-height: 1;
}
@media (min-width: 768px) {
    .metric-tile-value { font-size: 1.25rem; }
}
.metric-tile-value.green { color: var(--green); }
.metric-tile-value.blue  { color: var(--blue);  }
.metric-tile-value.amber { color: var(--amber); }
.metric-tile-value.red   { color: var(--red);   }
.metric-tile-sub { font-size: 0.65rem; color: var(--text-muted); margin-top: -4px; }

/* ═══════════════════════════════════════════════════════════
   DIVIDER
═══════════════════════════════════════════════════════════ */
.hdivider {
    height: 1px;
    background: linear-gradient(90deg,
        transparent 0%, var(--border-subtle) 30%,
        var(--border-subtle) 70%, transparent 100%);
    margin: 16px 0; border: none;
}

/* ═══════════════════════════════════════════════════════════
   CLASSES PANEL
═══════════════════════════════════════════════════════════ */
details {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--r-lg) !important;
    overflow: hidden; margin-bottom: 16px !important;
}
details summary {
    padding: 12px 16px !important;
    font-size: 0.75rem !important; font-weight: 600 !important;
    color: var(--text-secondary) !important; cursor: pointer;
    user-select: none; letter-spacing: 0.01em;
}
details summary:hover { color: var(--text-primary) !important; }
details[open] summary { border-bottom: 1px solid var(--border-faint); }
.class-pill {
    display: inline-flex; align-items: center; gap: 4px;
    background: var(--bg-elevated); border: 1px solid var(--border-faint);
    padding: 3px 9px; border-radius: var(--r-xs);
    font-size: 0.68rem; font-weight: 500; color: var(--text-secondary);
    font-family: 'Courier New', monospace; margin: 2px;
    transition: border-color var(--dur) var(--ease);
}
.class-pill:hover { border-color: var(--border-default); }
.class-pill.person { border-color: var(--red-border); color: var(--red); background: var(--red-dim); }

/* ═══════════════════════════════════════════════════════════
   ACTION BAR
═══════════════════════════════════════════════════════════ */
.action-bar {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 12px; flex-wrap: wrap; gap: 8px;
}
.action-bar-left { display: flex; flex-direction: column; gap: 2px; }
.action-bar-title {
    font-size: 0.85rem; font-weight: 600;
    color: var(--text-primary); letter-spacing: -0.01em;
}
.action-bar-sub { font-size: 0.70rem; color: var(--text-tertiary); }

/* ═══════════════════════════════════════════════════════════
   STREAMLIT BUTTON OVERRIDES
═══════════════════════════════════════════════════════════ */
div[data-testid="stButton"] > button {
    font-weight: 600 !important; font-size: 0.82rem !important;
    letter-spacing: 0.01em !important; border-radius: var(--r-md) !important;
    height: 42px !important; border: none !important;
    transition: all var(--dur) var(--ease) !important;
    white-space: nowrap !important; width: 100% !important;
}
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, var(--green) 0%, #00b36a 100%) !important;
    color: #05120d !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
}
div[data-testid="stButton"] > button[kind="secondary"] {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-default) !important;
    color: var(--text-secondary) !important;
}
div[data-testid="stButton"] > button[kind="secondary"]:hover {
    border-color: var(--border-strong) !important;
    color: var(--text-primary) !important;
    transform: translateY(-1px) !important;
}

/* ═══════════════════════════════════════════════════════════
   HERO / IDLE STATE — responsive padding
═══════════════════════════════════════════════════════════ */
.hero-wrap {
    background: var(--bg-card); border: 1px solid var(--border-subtle);
    border-radius: var(--r-xl); padding: 40px 20px 36px;
    text-align: center; position: relative; overflow: hidden;
    min-height: 320px; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
}
@media (min-width: 768px) {
    .hero-wrap { padding: 72px 40px 64px; min-height: 420px; }
}
.hero-wrap::after {
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(ellipse 60% 50% at 50% 0%,
        rgba(0,212,126,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-icon-ring {
    width: 70px; height: 70px; margin: 0 auto 20px;
    background: var(--bg-elevated); border: 1px solid var(--border-default);
    border-radius: 20px; display: flex; align-items: center; justify-content: center;
    position: relative; z-index: 1; font-size: 28px;
}
.hero-heading {
    font-size: 1.3rem; font-weight: 700; color: var(--text-primary);
    letter-spacing: -0.03em; margin-bottom: 10px; position: relative; z-index: 1;
}
@media (min-width: 768px) {
    .hero-heading { font-size: 1.55rem; }
}
.hero-body {
    font-size: 0.82rem; color: var(--text-tertiary); line-height: 1.7;
    max-width: 460px; margin: 0 auto 24px; position: relative; z-index: 1;
}
.hero-chips {
    display: flex; flex-wrap: wrap; justify-content: center;
    gap: 6px; position: relative; z-index: 1;
}
.h-chip {
    display: inline-flex; align-items: center; gap: 5px;
    background: var(--bg-elevated); border: 1px solid var(--border-subtle);
    border-radius: var(--r-sm); padding: 6px 11px;
    font-size: 0.70rem; font-weight: 500; color: var(--text-secondary);
    transition: all var(--dur) var(--ease);
}
.h-chip:hover {
    border-color: var(--border-default); color: var(--text-primary);
    transform: translateY(-1px);
}

/* ═══════════════════════════════════════════════════════════
   STATS SIDEBAR PANELS
═══════════════════════════════════════════════════════════ */
.sp {
    background: var(--bg-card); border: 1px solid var(--border-subtle);
    border-radius: var(--r-lg); padding: 14px 16px; margin-bottom: 10px;
    transition: border-color var(--dur) var(--ease);
}
.sp:hover { border-color: var(--border-default); }
.sp-head {
    display: flex; align-items: center; gap: 8px;
    margin-bottom: 12px; padding-bottom: 10px; border-bottom: 1px solid var(--border-faint);
}
.sp-icon {
    width: 24px; height: 24px; border-radius: var(--r-sm);
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; flex-shrink: 0;
}
.sp-icon.green  { background:var(--green-dim);  color:var(--green);  }
.sp-icon.blue   { background:var(--blue-dim);   color:var(--blue);   }
.sp-icon.red    { background:var(--red-dim);    color:var(--red);    }
.sp-icon.violet { background:var(--violet-dim); color:var(--violet); }
.sp-icon.amber  { background:var(--amber-dim);  color:var(--amber);  }
.sp-title {
    font-size: 0.67rem; font-weight: 700; color: var(--text-tertiary);
    text-transform: uppercase; letter-spacing: 0.07em;
}
.fps-big { display: flex; align-items: baseline; gap: 5px; margin-bottom: 4px; }
.fps-num {
    font-size: 2.2rem; font-weight: 800; color: var(--green);
    font-family: 'Courier New', monospace; letter-spacing: -0.04em; line-height: 1;
}
.fps-unit {
    font-size: 0.70rem; font-weight: 600; color: var(--text-tertiary);
    text-transform: uppercase; letter-spacing: 0.06em;
}
.fps-sub { font-size: 0.66rem; color: var(--text-muted); font-family: 'Courier New', monospace; }

/* Alert badges */
.alert-live {
    background: var(--red-dim); border: 1px solid var(--red-border);
    border-radius: var(--r-md); padding: 10px 13px;
    display: flex; align-items: center; gap: 10px;
    animation: breathe-red 2s ease-in-out infinite;
}
@keyframes breathe-red {
    0%,100% { box-shadow: 0 0 10px var(--red-glow); }
    50% { box-shadow: 0 0 22px var(--red-glow); }
}
.alert-live .adot {
    width: 7px; height: 7px; border-radius: 50%; background: var(--red);
    animation: pulse-dot 1.2s ease-in-out infinite; flex-shrink: 0;
}
.alert-live .atxt { font-size: 0.74rem; font-weight: 700; color: var(--red); letter-spacing: 0.04em; }
.alert-clear {
    background: var(--green-dim); border: 1px solid var(--green-border);
    border-radius: var(--r-md); padding: 10px 13px;
    display: flex; align-items: center; gap: 10px;
}
.alert-clear .adot { width: 7px; height: 7px; border-radius: 50%; background: var(--green); flex-shrink: 0; }
.alert-clear .atxt { font-size: 0.74rem; font-weight: 600; color: var(--green); }

/* Detection list */
.det-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 6px 0; border-bottom: 1px solid var(--border-faint);
}
.det-row:last-child { border-bottom: none; }
.det-left { display: flex; align-items: center; gap: 8px; }
.det-marker { width: 5px; height: 5px; border-radius: 50%; flex-shrink: 0; }
.det-marker.person { background: var(--red); }
.det-marker.obj    { background: var(--blue); }
.det-label { font-size: 0.76rem; font-weight: 500; color: var(--text-primary); text-transform: capitalize; }
.det-badge {
    font-size: 0.66rem; font-weight: 700; color: var(--text-secondary);
    background: var(--bg-elevated); border: 1px solid var(--border-faint);
    padding: 2px 8px; border-radius: 99px; font-family: 'Courier New', monospace;
}
.det-empty {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; gap: 6px; padding: 16px 0 6px;
    color: var(--text-muted); font-size: 0.73rem;
}
.det-empty .det-icon { font-size: 20px; opacity: 0.4; }

/* Session rows */
.srow {
    display: flex; justify-content: space-between; align-items: center;
    padding: 5px 0; border-bottom: 1px solid var(--border-faint);
}
.srow:last-child { border-bottom: none; }
.srow-label { font-size: 0.71rem; color: var(--text-tertiary); }
.srow-value {
    font-size: 0.74rem; font-weight: 700; color: var(--text-primary);
    font-family: 'Courier New', monospace;
}

/* ═══════════════════════════════════════════════════════════
   SIDEBAR — responsive
═══════════════════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border-faint) !important;
}
section[data-testid="stSidebar"] > div { padding: 16px 14px !important; }

.sb-heading {
    display: flex; align-items: center; gap: 7px;
    font-size: 0.64rem; font-weight: 700; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.09em;
    margin-bottom: 10px; padding-bottom: 8px;
    border-bottom: 1px solid var(--border-faint);
}
.sysbox {
    background: var(--bg-overlay); border: 1px solid var(--border-faint);
    border-radius: var(--r-md); padding: 12px 14px;
}
.sysrow {
    display: flex; justify-content: space-between; align-items: center;
    padding: 5px 0; border-bottom: 1px solid var(--border-faint);
}
.sysrow:last-child { border-bottom: none; }
.sysrow-lbl { font-size: 0.70rem; color: var(--text-tertiary); }
.sysrow-val { font-size: 0.70rem; font-weight: 600; color: var(--text-primary); font-family: 'Courier New', monospace; }
.chip-sm {
    display: inline-flex; align-items: center; gap: 3px;
    padding: 2px 8px; border-radius: var(--r-xs);
    font-size: 0.65rem; font-weight: 700; font-family: 'Courier New', monospace;
    letter-spacing: 0.04em;
}
.chip-sm.gpu { background:var(--green-dim); color:var(--green); }
.chip-sm.cpu { background:var(--amber-dim); color:var(--amber); }

label[data-testid="stWidgetLabel"] p {
    font-size: 0.75rem !important; font-weight: 500 !important;
    color: var(--text-secondary) !important;
}

/* ═══════════════════════════════════════════════════════════
   HIDE STREAMLIT CHROME
═══════════════════════════════════════════════════════════ */
#MainMenu, footer,
div[data-testid="stDecoration"],
div[data-testid="stStatusWidget"],
div[data-testid="metric-container"] {
    display: none !important; visibility: hidden !important;
}
/* Hide deploy link specifically, not the whole toolbar */
div[data-testid="stToolbar"] a[href*="deploy"] {
    display: none !important;
}
/* Make header blend with background */
header {
    background: var(--bg-base) !important;
    border-bottom: 1px solid var(--border-faint) !important;
}

/* ═══════════════════════════════════════════════════════════
   STOP BUTTON AREA
═══════════════════════════════════════════════════════════ */
.stop-row {
    display: flex; justify-content: flex-end;
    margin-top: 12px;
}
</style>
""", unsafe_allow_html=True)



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



_RNG = np.random.default_rng(42)
_CLASS_COLORS = _RNG.integers(80, 220, size=(300, 3), dtype=np.uint8)
_PERSON_COLOR = (71, 68, 239)
_SPECIAL_COLORS = {
    "car":        (246, 130,  59),
    "truck":      (200, 100,  50),
    "motorcycle": ( 11, 158, 245),
    "bicycle":    (126, 212,   0),
    "dog":        (126, 232, 100),
    "cat":        (200, 232, 100),
    "bird":       (255, 200, 100),
    "cell phone": (246,  92, 139),
    "laptop":     (246,  92, 200),
}


def _get_color(class_name: str, class_id: int) -> tuple:
    if class_name.lower() == "person":
        return _PERSON_COLOR
    if class_name.lower() in _SPECIAL_COLORS:
        return _SPECIAL_COLORS[class_name.lower()]
    c = _CLASS_COLORS[class_id % len(_CLASS_COLORS)]
    return int(c[0]), int(c[1]), int(c[2])


def detect_objects(frame, model, confidence_threshold=0.5, iou_threshold=0.45):
    frame = to_cpu_mat(frame)
    detections = []
    detection_counts = collections.Counter()
    human_detected = False

    results = model(
        frame.copy(), conf=confidence_threshold,
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
    return to_cpu_mat(annotated), detections, detection_counts, human_detected


def draw_detections(frame, detections, human_detected=False):
    h, w = frame.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        name = det["class"]
        conf = det["confidence"]
        cid  = det["class_id"]
        color = _get_color(name, cid)
        is_person = name.lower() == "person"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cl, ct = 14, 3
        for px, py, sx, sy in [
            (x1, y1, 1, 1), (x2, y1, -1, 1),
            (x1, y2, 1, -1), (x2, y2, -1, -1),
        ]:
            cv2.line(frame, (px, py), (px + sx*cl, py), color, ct)
            cv2.line(frame, (px, py), (px, py + sy*cl), color, ct)

        label = f"{name}  {conf:.0%}"
        font  = cv2.FONT_HERSHEY_SIMPLEX
        fs, ft = 0.46, 1
        (tw, th), bl = cv2.getTextSize(label, font, fs, ft)
        ly  = max(y1 - 6, th + 8)
        px2, py2 = 6, 4
        cv2.rectangle(frame,
            (x1, ly - th - py2*2), (x1 + tw + px2*2, ly + 2), color, -1)
        cv2.putText(frame, label,
            (x1 + px2, ly - py2), font, fs, (255, 255, 255), ft, cv2.LINE_AA)

        cx, cy = (x1+x2)//2, (y1+y2)//2
        cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA)

        if is_person:
            bh = y2 - y1
            if bh > 10:
                cv2.putText(frame,
                    f"~{max(1,int(500/bh))}m",
                    (x1, y2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (180, 200, 220), 1, cv2.LINE_AA)

    bar_h = 34
    ov = frame.copy()
    cv2.rectangle(ov, (0, h - bar_h), (w, h), (8, 8, 14), -1)
    cv2.addWeighted(ov, 0.72, frame, 0.28, 0, frame)
    cv2.putText(frame,
        f"{len(detections)} object{'s' if len(detections)!=1 else ''} detected",
        (12, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
        (150, 160, 175), 1, cv2.LINE_AA)

    if human_detected:
        alert = "HUMAN DETECTED"
        (aw, _), _ = cv2.getTextSize(alert, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1)
        cv2.putText(frame, alert,
            (w - aw - 12, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
            (71, 68, 239), 1, cv2.LINE_AA)
    return frame



def frame_to_pil(frame):
    frame = to_cpu_mat(frame)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8))



class FPSCounter:
    def __init__(self, n=30):
        self._ts = collections.deque(maxlen=n)

    def tick(self):
        self._ts.append(time.perf_counter())

    def fps(self):
        if len(self._ts) < 2:
            return 0.0
        dt = self._ts[-1] - self._ts[0]
        return 0.0 if dt == 0 else (len(self._ts) - 1) / dt



if _MP_OK:
    try:
        _mp_hands = mp.solutions.hands
        _mp_draw  = mp.solutions.drawing_utils
        _mp_style = mp.solutions.drawing_styles
    except AttributeError:
        _MP_OK = False
        _mp_hands = None
        _mp_draw  = None
        _mp_style = None
else:
    _mp_hands = None
    _mp_draw  = None
    _mp_style = None

@st.cache_resource
def _get_hand_detector(max_hands=2, min_conf=0.6):
    return _mp_hands.Hands(
        static_image_mode=False, max_num_hands=max_hands,
        min_detection_confidence=min_conf, min_tracking_confidence=min_conf,
    )

_WRIST = 0
_THUMB_TIP  = 4;  _THUMB_IP  = 3;  _THUMB_MCP = 2
_INDEX_TIP  = 8;  _INDEX_PIP = 6
_MIDDLE_TIP = 12; _MIDDLE_PIP= 10
_RING_TIP   = 16; _RING_PIP  = 14
_PINKY_TIP  = 20; _PINKY_PIP = 18

def _finger_up(lm, tip, pip, wrist=_WRIST):
    return lm[tip].y < lm[pip].y

def _thumb_up(lm, handedness):
    if handedness == "Right":
        return lm[_THUMB_TIP].x < lm[_THUMB_IP].x
    return lm[_THUMB_TIP].x > lm[_THUMB_IP].x

def classify_gesture(lm, handedness) -> str:
    thumb  = _thumb_up(lm, handedness)
    index  = _finger_up(lm, _INDEX_TIP, _INDEX_PIP)
    middle = _finger_up(lm, _MIDDLE_TIP, _MIDDLE_PIP)
    ring   = _finger_up(lm, _RING_TIP, _RING_PIP)
    pinky  = _finger_up(lm, _PINKY_TIP, _PINKY_PIP)
    fingers = [thumb, index, middle, ring, pinky]
    count   = sum(fingers)
    if count == 0: return "Fist"
    if count == 5: return "Open Palm"
    if thumb and not index and not middle and not ring and not pinky: return "Thumbs Up"
    if not thumb and index and middle and not ring and not pinky: return "Peace"
    if not thumb and index and not middle and not ring and not pinky: return "Pointing"
    if not thumb and index and not middle and not ring and pinky: return "Rock"
    if thumb and index and not middle and not ring and pinky: return "Hang Loose"
    if not thumb and not index and not middle and not ring and pinky: return "Pinky"
    if not thumb and index and middle and ring and not pinky: return "Three"
    if not thumb and index and middle and ring and pinky: return "Four"
    return f"{count} Fingers"

if _MP_OK:
    try:
        _HAND_CONNECTIONS = _mp_hands.HAND_CONNECTIONS
        _HAND_LANDMARK_STYLE = _mp_style.get_default_hand_landmarks_style()
        _HAND_CONN_STYLE = _mp_style.get_default_hand_connections_style()
    except AttributeError:
        _MP_OK = False
        _HAND_CONNECTIONS = None
        _HAND_LANDMARK_STYLE = None
        _HAND_CONN_STYLE = None
else:
    _HAND_CONNECTIONS = None
    _HAND_LANDMARK_STYLE = None
    _HAND_CONN_STYLE = None

def detect_hands(frame, hand_detector):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hand_detector.process(rgb)
    gestures = []
    if results.multi_hand_landmarks and results.multi_handedness:
        for hlm, hclass in zip(results.multi_hand_landmarks, results.multi_handedness):
            handedness = hclass.classification[0].label
            gesture = classify_gesture(hlm.landmark, handedness)
            gestures.append({"gesture": gesture, "handedness": handedness, "landmarks": hlm})
    return gestures

def draw_hands(frame, gestures):
    for g in gestures:
        hlm = g["landmarks"]
        _mp_draw.draw_landmarks(frame, hlm, _HAND_CONNECTIONS,
            _HAND_LANDMARK_STYLE, _HAND_CONN_STYLE)
        h, w = frame.shape[:2]
        wrist = hlm.landmark[_WRIST]
        cx, cy = int(wrist.x * w), int(wrist.y * h)
        label = f'{g["gesture"]} ({g["handedness"]})'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (cx, cy - th - 14), (cx + tw + 8, cy - 4), (34, 211, 238), -1)
        cv2.putText(frame, label, (cx + 4, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


if _MP_OK:
    try:
        _mp_face = mp.solutions.face_mesh
        _FACE_CONTOURS = _mp_face.FACEMESH_CONTOURS
    except AttributeError:
        _MP_OK = False
        _mp_face = None
        _FACE_CONTOURS = None
else:
    _mp_face = None
    _FACE_CONTOURS = None

@st.cache_resource
def _get_face_detector(min_conf=0.5):
    return _mp_face.FaceMesh(
        static_image_mode=False, max_num_faces=2,
        refine_landmarks=True, min_detection_confidence=min_conf,
        min_tracking_confidence=min_conf,
    )

_L_EYE_TOP = 159; _L_EYE_BOT = 145
_R_EYE_TOP = 386; _R_EYE_BOT = 374
_L_EYE_IN  = 133; _L_EYE_OUT = 33
_R_EYE_IN  = 362; _R_EYE_OUT = 263
_L_BROW_IN = 70;  _L_BROW_OUT = 63
_R_BROW_IN = 300; _R_BROW_OUT = 293
_LIP_TOP   = 13;  _LIP_BOT    = 14
_LIP_L     = 61;  _LIP_R      = 291
_NOSE_TIP  = 1;   _CHIN       = 152
_L_MOUTH   = 61;  _R_MOUTH    = 291

def _dist(lm, a, b):
    return ((lm[a].x - lm[b].x)**2 + (lm[a].y - lm[b].y)**2)**0.5

def classify_expression(lm) -> str:
    l_eye_h = _dist(lm, _L_EYE_TOP, _L_EYE_BOT)
    r_eye_h = _dist(lm, _R_EYE_TOP, _R_EYE_BOT)
    l_eye_w = _dist(lm, _L_EYE_IN, _L_EYE_OUT)
    r_eye_w = _dist(lm, _R_EYE_IN, _R_EYE_OUT)
    eye_ratio = ((l_eye_h / max(l_eye_w, 1e-6)) + (r_eye_h / max(r_eye_w, 1e-6))) / 2
    l_brow_h = _dist(lm, _L_BROW_OUT, _L_EYE_TOP)
    r_brow_h = _dist(lm, _R_BROW_OUT, _R_EYE_TOP)
    face_h   = _dist(lm, _NOSE_TIP, _CHIN)
    brow_ratio = ((l_brow_h + r_brow_h) / 2) / max(face_h, 1e-6)
    mouth_h = _dist(lm, _LIP_TOP, _LIP_BOT)
    mouth_w = _dist(lm, _LIP_L, _LIP_R)
    mouth_ratio = mouth_h / max(mouth_w, 1e-6)
    lip_top_y  = lm[_LIP_TOP].y; lip_bot_y = lm[_LIP_BOT].y
    lip_l_y    = lm[_L_MOUTH].y; lip_r_y   = lm[_R_MOUTH].y
    corner_avg = (lip_l_y + lip_r_y) / 2
    lip_center = (lip_top_y + lip_bot_y) / 2
    smile_score = lip_center - corner_avg
    if eye_ratio < 0.08: return "Eyes Closed"
    if mouth_ratio > 0.55:
        if smile_score > 0.005: return "Laughing"
        return "Mouth Open"
    if smile_score > 0.008 and mouth_ratio < 0.35: return "Smiling"
    if brow_ratio > 0.35: return "Surprised"
    if brow_ratio < 0.18: return "Angry"
    if eye_ratio < 0.15: return "Sleepy"
    return "Neutral"

def detect_faces(frame, face_detector):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb)
    faces = []
    if results.multi_face_landmarks:
        for flm in results.multi_face_landmarks:
            expr = classify_expression(flm.landmark)
            faces.append({"expression": expr, "landmarks": flm})
    return faces

def draw_faces(frame, faces):
    for f in faces:
        flm = f["landmarks"]
        _mp_draw.draw_landmarks(
            frame, flm, _FACE_CONTOURS,
            landmark_drawing_spec=_mp_draw.DrawingSpec(color=(200, 200, 255), thickness=1, circle_radius=1),
            connection_drawing_spec=_mp_draw.DrawingSpec(color=(100, 100, 160), thickness=1),
        )
        h, w = frame.shape[:2]
        nose = flm.landmark[_NOSE_TIP]
        cx, cy = int(nose.x * w), int(nose.y * h)
        label = f["expression"]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (cx - tw//2 - 4, cy - th - 16), (cx + tw//2 + 4, cy - 6), (200, 130, 255), -1)
        cv2.putText(frame, label, (cx - tw//2, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame



_INDEX_MCP  = 5
_MIDDLE_MCP = 9
_RING_MCP   = 13
_PINKY_MCP  = 17

def _palm_center(lm, w, h):
    ids = [_WRIST, _INDEX_MCP, _MIDDLE_MCP, _RING_MCP, _PINKY_MCP]
    cx = sum(lm[i].x for i in ids) / len(ids) * w
    cy = sum(lm[i].y for i in ids) / len(ids) * h
    return cx, cy

def _pointing_dir(lm, w, h):
    dx = (lm[_INDEX_TIP].x - lm[_WRIST].x) * w
    dy = (lm[_INDEX_TIP].y - lm[_WRIST].y) * h
    mag = max((dx**2 + dy**2)**0.5, 1e-6)
    return dx / mag, dy / mag


class ParticleSystem:
    def __init__(self, max_particles=200):
        self.max_n = max_particles
        self.px    = np.zeros(max_particles, dtype=np.float32)
        self.py    = np.zeros(max_particles, dtype=np.float32)
        self.vx    = np.zeros(max_particles, dtype=np.float32)
        self.vy    = np.zeros(max_particles, dtype=np.float32)
        self.life  = np.zeros(max_particles, dtype=np.float32)
        self.color = np.zeros((max_particles, 3), dtype=np.float32)
        self.alive = np.zeros(max_particles, dtype=bool)

    @property
    def count(self):
        return int(self.alive.sum())

    def spawn(self, x, y, vx, vy, color, n=1):
        dead = np.where(~self.alive)[0]
        n = min(n, len(dead))
        if n == 0:
            return
        idx = dead[:n]
        self.px[idx]    = x + np.random.randn(n).astype(np.float32) * 4
        self.py[idx]    = y + np.random.randn(n).astype(np.float32) * 4
        self.vx[idx]    = vx + np.random.randn(n).astype(np.float32) * 0.8
        self.vy[idx]    = vy + np.random.randn(n).astype(np.float32) * 0.8
        self.life[idx]  = 1.0
        self.color[idx] = color
        self.alive[idx] = True

    def update(self, dt, gesture=None, cx=None, cy=None,
               dir_x=None, dir_y=None, fw=640, fh=480):
        mask = self.alive
        if not mask.any():
            return
        if gesture and cx is not None and cy is not None:
            dx = self.px[mask] - cx
            dy = self.py[mask] - cy
            dist = np.sqrt(dx**2 + dy**2)
            dist = np.maximum(dist, 10.0)
            nx = dx / dist; ny = dy / dist
            if gesture == "Fist":
                force = 180.0 / dist * dt
                self.vx[mask] -= nx * force; self.vy[mask] -= ny * force
            elif gesture == "Open Palm":
                force = 600.0 / (dist + 40) * dt
                self.vx[mask] += nx * force; self.vy[mask] += ny * force
            elif gesture == "Pointing" and dir_x is not None:
                force = 120.0 * dt
                self.vx[mask] += dir_x * force; self.vy[mask] += dir_y * force
                self.vy[mask] -= dy * 0.5 * dt
            elif gesture == "Thumbs Up":
                self.vy[mask] -= 200.0 * dt
                self.vx[mask] += np.random.randn(int(mask.sum())).astype(np.float32) * 30 * dt
            elif gesture == "Hang Loose":
                tx = -ny; ty = nx
                force = 100.0 * dt
                self.vx[mask] += tx * force; self.vy[mask] += ty * force
                self.vx[mask] += nx * 15 * dt; self.vy[mask] += ny * 15 * dt

        self.vy[mask] += 30.0 * dt
        self.vx[mask] *= 0.98; self.vy[mask] *= 0.98
        self.px[mask] += self.vx[mask] * dt
        self.py[mask] += self.vy[mask] * dt
        self.life[mask] -= dt * 0.5
        dead = (self.life <= 0) | (self.px < -20) | (self.px > fw + 20) | \
               (self.py < -20) | (self.py > fh + 20)
        self.alive[dead] = False

    def render(self, frame):
        """
        FIX 4 — Particles are now drawn at full opacity directly onto the frame
        (no addWeighted blend that was washing them out).  Each particle gets:
          • a filled solid circle (radius 4–8 based on life)
          • a 1-px outline ring for a 'neon' glow look without actual blur
        This makes them clearly visible against any real-world background.
        """
        mask = self.alive
        if not mask.any():
            return frame
        idxs = np.where(mask)[0]
        xs    = self.px[idxs].astype(int)
        ys    = self.py[idxs].astype(int)
        lives = self.life[idxs]
        colors = self.color[idxs]

        for i in range(len(idxs)):
            a = float(np.clip(lives[i], 0.0, 1.0))
            # Core radius scales with life: 4 at birth → 2 near death
            r_core = max(2, int(4 * a))
            # Outer glow ring radius
            r_glow = r_core + 3

            # Full-brightness core color
            c = (int(colors[i][0]), int(colors[i][1]), int(colors[i][2]))
            # Dimmed outer ring (50% brightness)
            c_dim = (c[0] // 2, c[1] // 2, c[2] // 2)

            x, y = xs[i], ys[i]
            # Draw glow ring first (behind core)
            cv2.circle(frame, (x, y), r_glow, c_dim, 1, cv2.LINE_AA)
            # Draw solid core on top — full opacity, no blending
            cv2.circle(frame, (x, y), r_core, c, -1, cv2.LINE_AA)
            # Bright center highlight for contrast
            if r_core >= 3:
                cv2.circle(frame, (x, y), max(1, r_core - 2), (255, 255, 255), -1, cv2.LINE_AA)

        return frame


_GESTURE_COLORS = {
    "Fist":       np.array([34, 211, 238], dtype=np.float32),
    "Open Palm":  np.array([71, 68, 239], dtype=np.float32),
    "Pointing":   np.array([0, 212, 126], dtype=np.float32),
    "Peace":      np.array([232, 168, 56], dtype=np.float32),
    "Thumbs Up":  np.array([56, 180, 232], dtype=np.float32),
    "Hang Loose": np.array([180, 56, 232], dtype=np.float32),
    "Rock":       np.array([232, 56, 180], dtype=np.float32),
}

def _spawn_for_gesture(ps, gesture, cx, cy, dir_x=None, dir_y=None):
    color = _GESTURE_COLORS.get(gesture, np.array([200, 200, 200], dtype=np.float32))
    if gesture == "Fist":
        ps.spawn(cx, cy, 0, 0, color, n=4)
    elif gesture == "Open Palm":
        for _ in range(6):
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(60, 200)
            ps.spawn(cx, cy, np.cos(angle)*speed, np.sin(angle)*speed, color, n=1)
    elif gesture == "Pointing" and dir_x is not None:
        ps.spawn(cx, cy, dir_x * 150, dir_y * 150, color, n=3)
    elif gesture == "Peace" and dir_x is not None:
        ps.spawn(cx, cy, dir_x * 120, dir_y * 120, color, n=2)
        ps.spawn(cx, cy, dir_x * 80, dir_y * 80, color, n=2)
    elif gesture == "Thumbs Up":
        for _ in range(3):
            ps.spawn(cx + np.random.randn()*8, cy,
                      np.random.randn()*20, -np.random.uniform(80, 200), color, n=1)
    elif gesture == "Hang Loose":
        angle = np.random.uniform(0, 2 * np.pi)
        ps.spawn(cx, cy, np.cos(angle)*60, np.sin(angle)*60, color, n=3)
    elif gesture == "Rock":
        ps.spawn(cx, cy, 0, 0, color, n=2)
    else:
        ps.spawn(cx, cy, np.random.randn()*10, np.random.randn()*10, color, n=1)



def _render_image(placeholder, img, caption=""):
    if "_img_style" not in st.session_state:
        import inspect
        p = inspect.signature(st.image).parameters
        if "use_container_width" in p:
            st.session_state["_img_style"] = "old"
        else:
            st.session_state["_img_style"] = "plain"
    style = st.session_state["_img_style"]
    if style == "old":
        placeholder.image(img, use_container_width=True, caption=caption)
    else:
        placeholder.image(img, caption=caption)


import subprocess
import platform

def _init_tts():
    pass

def _speak(text):
    try:
        if platform.system() == "Windows":
            cmd = f'powershell -Command "Add-Type -AssemblyName System.Speech; $s=New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak(\'{text}\')"'
            subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

def render_sidebar():
    st.sidebar.markdown("""
    <div style="display:flex;align-items:center;gap:10px;
                padding:0 2px 16px;
                border-bottom:1px solid var(--border-faint);
                margin-bottom:16px;">
        <div style="width:30px;height:30px;
                    background:linear-gradient(145deg,#00d47e,#4f8ef7);
                    border-radius:9px;display:flex;
                    align-items:center;justify-content:center;
                    flex-shrink:0;font-size:14px;color:#fff;">◉</div>
        <div>
            <div style="font-size:0.85rem;font-weight:700;
                        color:var(--text-primary);letter-spacing:-0.02em;">
                Live Object Detection & <span style="color:var(--green);">Tracing</span>
            </div>
            <div style="font-size:0.60rem;color:var(--text-muted);
                        letter-spacing:0.04em;">Configuration Panel</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown('<div class="sb-heading">⬡ Model</div>', unsafe_allow_html=True)
    model_map = {
        "Nano  — Fastest":       "yolov8n",
        "Small — Balanced":      "yolov8s",
        "Medium — Accurate":     "yolov8m",
        "Large — High Accuracy": "yolov8l",
        "XLarge — Maximum":      "yolov8x",
    }
    label = st.sidebar.selectbox("Variant", list(model_map.keys()), index=0, label_visibility="collapsed")

    st.sidebar.markdown('<div class="sb-heading" style="margin-top:16px;">≡ Detection Thresholds</div>', unsafe_allow_html=True)
    conf = st.sidebar.slider("Confidence threshold", 0.10, 1.0, 0.50, 0.05)
    iou  = st.sidebar.slider("NMS IoU threshold", 0.10, 1.0, 0.45, 0.05)

    st.sidebar.markdown('<div class="sb-heading" style="margin-top:16px;">◎ Camera</div>', unsafe_allow_html=True)
    cam_source = st.sidebar.radio("Source", ["Local Camera", "Browser Webcam"], index=1, horizontal=True, help="Browser Webcam works on hosted sites")
    cam_idx = st.sidebar.number_input("Device index", min_value=0, max_value=10, value=0, disabled=(cam_source == "Browser Webcam"))
    res_map = {
        "640 × 480":  (640, 480),
        "800 × 600":  (800, 600),
        "1280 × 720": (1280, 720),
    }
    res_lbl    = st.sidebar.selectbox("Resolution", list(res_map.keys()), index=0)
    resolution = res_map[res_lbl]

    st.sidebar.markdown('<div class="sb-heading" style="margin-top:16px;">⚙ Display</div>', unsafe_allow_html=True)
    show_fps = st.sidebar.checkbox("FPS overlay", value=True)
    flip     = st.sidebar.checkbox("Mirror camera", value=True)
    max_fps  = st.sidebar.slider("Frame-rate cap", 5, 60, 30)

    st.sidebar.markdown('<div class="sb-heading" style="margin-top:16px;">✋ Hand Gestures</div>', unsafe_allow_html=True)
    hand_detect = st.sidebar.checkbox("Enable hand detection", value=False, disabled=not _MP_OK)
    hand_conf   = st.sidebar.slider("Hand confidence", 0.30, 1.0, 0.60, 0.05, disabled=not hand_detect)
    if not _MP_OK:
        st.sidebar.caption("Install mediapipe to enable")

    st.sidebar.markdown('<div class="sb-heading" style="margin-top:16px;">☺ Face Expressions</div>', unsafe_allow_html=True)
    face_detect = st.sidebar.checkbox("Enable face detection", value=False, disabled=not _MP_OK)
    face_conf   = st.sidebar.slider("Face confidence", 0.30, 1.0, 0.50, 0.05, disabled=not face_detect)
    if not _MP_OK:
        st.sidebar.caption("Install mediapipe to enable")

    st.sidebar.markdown('<div class="sb-heading" style="margin-top:16px;">✦ Particle Effects</div>', unsafe_allow_html=True)
    particle_fx = st.sidebar.checkbox("Enable particle effects", value=False)
    particle_n  = st.sidebar.slider("Max particles", 50, 400, 200, 50, disabled=not particle_fx)

    st.sidebar.markdown('<div class="sb-heading" style="margin-top:16px;">⬛ Save Frames</div>', unsafe_allow_html=True)
    save_frames = st.sidebar.checkbox("Save detected frames", value=False)
    save_folder = st.sidebar.text_input("Save folder path", value="./saved_frames", disabled=not save_frames)
    save_interval = st.sidebar.slider("Save every N frames", 1, 60, 10, 1, disabled=not save_frames, help="Higher values save fewer frames to reduce disk usage")
    max_saved = st.sidebar.slider("Max saved frames", 10, 200, 50, 10, disabled=not save_frames, help="Oldest frames auto-deleted when limit reached")

    st.sidebar.markdown('<div class="sb-heading" style="margin-top:16px;">◈ Audio Alerts</div>', unsafe_allow_html=True)
    audio_alerts = st.sidebar.checkbox("Enable text-to-speech alerts", value=False)

    st.sidebar.markdown('<div class="sb-heading" style="margin-top:16px;">ℹ System</div>', unsafe_allow_html=True)
    is_gpu   = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0)[:22] if is_gpu else "—"
    b_cls    = "gpu" if is_gpu else "cpu"
    b_txt    = "CUDA" if is_gpu else "CPU"
    st.sidebar.markdown(f"""
    <div class="sysbox">
        <div class="sysrow">
            <span class="sysrow-lbl">Compute</span>
            <span class="chip-sm {b_cls}">{b_txt}</span>
        </div>
        <div class="sysrow">
            <span class="sysrow-lbl">GPU</span>
            <span class="sysrow-val">{gpu_name}</span>
        </div>
        <div class="sysrow">
            <span class="sysrow-lbl">PyTorch</span>
            <span class="sysrow-val">{torch.__version__}</span>
        </div>
        <div class="sysrow">
            <span class="sysrow-lbl">OpenCL</span>
            <span class="sysrow-val">{"on" if cv2.ocl.useOpenCL() else "off"}</span>
        </div>
    </div>""", unsafe_allow_html=True)

    return {
        "model":        model_map[label],
        "confidence":   conf,
        "iou":          iou,
        "camera_index": int(cam_idx),
        "cam_source":   cam_source,
        "resolution":   resolution,
        "show_fps":     show_fps,
        "flip_camera":  flip,
        "max_fps":      max_fps,
        "hand_detect":  hand_detect,
        "hand_conf":    hand_conf,
        "face_detect":  face_detect,
        "face_conf":    face_conf,
        "particle_fx":  particle_fx,
        "particle_n":   particle_n,
        "save_frames":  save_frames,
        "save_folder":  save_folder,
        "save_interval": save_interval,
        "max_saved":     max_saved,
        "audio_alerts": audio_alerts,
    }


# Module-level vars for WebRTC callback (session_state is NOT thread-safe in callbacks)
_WEBRTC_MODEL = None
_WEBRTC_CONF = 0.5
_WEBRTC_IOU = 0.45
_WEBRTC_FLIP = True
_WEBRTC_SHOW_FPS = True
_WEBRTC_MODEL_NAME = "yolov8n"
_WEBRTC_SAVE = False
_WEBRTC_SAVE_FOLDER = "./saved_frames"
_WEBRTC_SAVE_INTERVAL = 10
_WEBRTC_MAX_SAVED = 50
_WEBRTC_FRAME_N = 0
_WEBRTC_TOTAL_DETS = 0
_WEBRTC_FPS = 0.0
_WEBRTC_T0 = 0.0
_WEBRTC_HAND_DETECT = False
_WEBRTC_HAND_CONF = 0.6
_WEBRTC_FACE_DETECT = False
_WEBRTC_FACE_CONF = 0.5
_WEBRTC_PARTICLE_FX = False
_WEBRTC_PARTICLE_N = 200
_WEBRTC_AUDIO_ALERTS = False
_WEBRTC_DET_COUNTS = None
_WEBRTC_HUMAN = False
_WEBRTC_GESTURES = []
_WEBRTC_FACES = []
_WEBRTC_HAND_DETECTOR = None
_WEBRTC_FACE_DETECTOR = None
_WEBRTC_PSYS = None
_WEBRTC_PREV_T = 0.0
_WEBRTC_LAST_HUMAN_ALERT = 0.0
_WEBRTC_LAST_GESTURE_ALERT = 0.0
_WEBRTC_LAST_EXPRESSION_ALERT = 0.0


def run_browser_detection(config, model):
    """Browser webcam mode - real-time WebRTC detection with full features"""
    global _WEBRTC_MODEL, _WEBRTC_CONF, _WEBRTC_IOU, _WEBRTC_FLIP, _WEBRTC_SHOW_FPS
    global _WEBRTC_MODEL_NAME, _WEBRTC_SAVE, _WEBRTC_SAVE_FOLDER, _WEBRTC_SAVE_INTERVAL
    global _WEBRTC_MAX_SAVED, _WEBRTC_FRAME_N, _WEBRTC_TOTAL_DETS, _WEBRTC_FPS, _WEBRTC_T0
    global _WEBRTC_HAND_DETECT, _WEBRTC_HAND_CONF, _WEBRTC_FACE_DETECT, _WEBRTC_FACE_CONF
    global _WEBRTC_PARTICLE_FX, _WEBRTC_PARTICLE_N, _WEBRTC_AUDIO_ALERTS
    global _WEBRTC_DET_COUNTS, _WEBRTC_HUMAN, _WEBRTC_GESTURES, _WEBRTC_FACES
    global _WEBRTC_HAND_DETECTOR, _WEBRTC_FACE_DETECTOR, _WEBRTC_PSYS, _WEBRTC_PREV_T
    global _WEBRTC_LAST_HUMAN_ALERT, _WEBRTC_LAST_GESTURE_ALERT, _WEBRTC_LAST_EXPRESSION_ALERT

    if not _WEBRTC_OK:
        error_msg = _WEBRTC_ERROR if "_WEBRTC_ERROR" in globals() else "Import failed"
        st.error(f"Browser webcam not available: {error_msg}")
        st.code("pip install streamlit-webrtc av", language="bash")
        return

    # Store config in module-level globals
    _WEBRTC_MODEL = model
    _WEBRTC_CONF = config["confidence"]
    _WEBRTC_IOU = config["iou"]
    _WEBRTC_FLIP = config["flip_camera"]
    _WEBRTC_SHOW_FPS = config["show_fps"]
    _WEBRTC_MODEL_NAME = config["model"].upper()
    _WEBRTC_SAVE = config["save_frames"]
    _WEBRTC_SAVE_FOLDER = config["save_folder"]
    _WEBRTC_SAVE_INTERVAL = config["save_interval"]
    _WEBRTC_MAX_SAVED = config["max_saved"]
    _WEBRTC_FRAME_N = 0
    _WEBRTC_TOTAL_DETS = 0
    _WEBRTC_FPS = 0.0
    _WEBRTC_T0 = time.perf_counter()
    _WEBRTC_HAND_DETECT = config["hand_detect"] and _MP_OK
    _WEBRTC_HAND_CONF = config["hand_conf"]
    _WEBRTC_FACE_DETECT = config["face_detect"] and _MP_OK
    _WEBRTC_FACE_CONF = config["face_conf"]

    # Warn if mediapipe not available but features enabled
    if config["hand_detect"] and not _MP_OK:
        mp_error = _MP_ERROR if "_MP_ERROR" in globals() else "Import failed"
        st.warning(f"Hand detection disabled: MediaPipe not available ({mp_error})")
    if config["face_detect"] and not _MP_OK:
        mp_error = _MP_ERROR if "_MP_ERROR" in globals() else "Import failed"
        st.warning(f"Face detection disabled: MediaPipe not available ({mp_error})")
    _WEBRTC_PARTICLE_FX = config["particle_fx"] and config["hand_detect"]
    _WEBRTC_PARTICLE_N = config["particle_n"]
    _WEBRTC_AUDIO_ALERTS = config["audio_alerts"]
    _WEBRTC_DET_COUNTS = collections.Counter()
    _WEBRTC_HUMAN = False
    _WEBRTC_GESTURES = []
    _WEBRTC_FACES = []
    _WEBRTC_LAST_HUMAN_ALERT = 0.0
    _WEBRTC_LAST_GESTURE_ALERT = 0.0
    _WEBRTC_LAST_EXPRESSION_ALERT = 0.0

    # Init hand/face detectors
    if _WEBRTC_HAND_DETECT:
        _WEBRTC_HAND_DETECTOR = _get_hand_detector(min_conf=_WEBRTC_HAND_CONF)
    else:
        _WEBRTC_HAND_DETECTOR = None
    if _WEBRTC_FACE_DETECT:
        _WEBRTC_FACE_DETECTOR = _get_face_detector(min_conf=_WEBRTC_FACE_CONF)
    else:
        _WEBRTC_FACE_DETECTOR = None
    if _WEBRTC_PARTICLE_FX:
        _WEBRTC_PSYS = ParticleSystem(max_particles=_WEBRTC_PARTICLE_N)
    else:
        _WEBRTC_PSYS = None
    _WEBRTC_PREV_T = time.perf_counter()

    # Init TTS
    if _WEBRTC_AUDIO_ALERTS:
        _init_tts()

    # Create save folder
    if _WEBRTC_SAVE:
        os.makedirs(_WEBRTC_SAVE_FOLDER, exist_ok=True)

    # Layout — same as local camera (no tabs during live)
    col_feed, col_panel = st.columns([3, 1], gap="large")

    with col_feed:
        st.markdown("""
        <div class="action-bar">
            <div class="action-bar-left">
                <div class="action-bar-title">Browser Webcam Feed</div>
                <div class="action-bar-sub">Real-time inference via browser webcam</div>
            </div>
        </div>""", unsafe_allow_html=True)

    with col_panel:
        st.markdown('<div style="height:48px;"></div>', unsafe_allow_html=True)
        fps_ph     = st.empty()
        alert_ph   = st.empty()
        detlist_ph = st.empty()
        gesture_ph = st.empty()
        expr_ph    = st.empty()
        session_ph = st.empty()

    def video_frame_callback(frame):
        global _WEBRTC_FRAME_N, _WEBRTC_TOTAL_DETS, _WEBRTC_FPS
        global _WEBRTC_DET_COUNTS, _WEBRTC_HUMAN, _WEBRTC_GESTURES, _WEBRTC_FACES
        global _WEBRTC_PREV_T, _WEBRTC_LAST_HUMAN_ALERT, _WEBRTC_LAST_GESTURE_ALERT
        global _WEBRTC_LAST_EXPRESSION_ALERT

        img = frame.to_ndarray(format="bgr24")

        # Flip if mirror enabled
        if _WEBRTC_FLIP:
            img = cv2.flip(img, 1)

        img = to_cpu_mat(img)

        # ── Object detection ─────────────────────────────
        try:
            ann, dets, det_counts, human = detect_objects(
                img, _WEBRTC_MODEL, _WEBRTC_CONF, _WEBRTC_IOU)
        except Exception:
            ann = img
            dets = []
            det_counts = collections.Counter()
            human = False

        _WEBRTC_DET_COUNTS = det_counts
        _WEBRTC_HUMAN = human

        # ── Hand detection ───────────────────────────────
        HAND_SKIP = 3
        if _WEBRTC_HAND_DETECTOR is not None and _WEBRTC_FRAME_N % HAND_SKIP == 0:
            try:
                small = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
                _WEBRTC_GESTURES = detect_hands(small, _WEBRTC_HAND_DETECTOR)
            except Exception:
                _WEBRTC_GESTURES = []
        gestures = _WEBRTC_GESTURES if _WEBRTC_HAND_DETECTOR is not None else []
        if gestures:
            ann = draw_hands(ann, gestures)

        # ── Face detection ───────────────────────────────
        FACE_SKIP = 4
        if _WEBRTC_FACE_DETECTOR is not None and _WEBRTC_FRAME_N % FACE_SKIP == 0:
            try:
                small = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
                _WEBRTC_FACES = detect_faces(small, _WEBRTC_FACE_DETECTOR)
            except Exception:
                _WEBRTC_FACES = []
        faces = _WEBRTC_FACES if _WEBRTC_FACE_DETECTOR is not None else []
        if faces:
            ann = draw_faces(ann, faces)

        # ── Particle effects ─────────────────────────────
        if _WEBRTC_PSYS is not None:
            now_t = time.perf_counter()
            dt = min(now_t - _WEBRTC_PREV_T, 0.1)
            _WEBRTC_PREV_T = now_t
            fh, fw = ann.shape[:2]
            if gestures:
                g = gestures[0]
                lm = g["landmarks"].landmark
                pcx, pcy = _palm_center(lm, fw, fh)
                gest = g["gesture"]
                dx, dy = None, None
                if gest in ("Pointing", "Peace"):
                    dx, dy = _pointing_dir(lm, fw, fh)
                _spawn_for_gesture(_WEBRTC_PSYS, gest, pcx, pcy, dx, dy)
                _WEBRTC_PSYS.update(dt, gest, pcx, pcy, dx, dy, fw, fh)
            else:
                _WEBRTC_PSYS.update(dt, fw=fw, fh=fh)
            ann = _WEBRTC_PSYS.render(ann)

        # ── Audio alerts ────────────────────────────────
        if _WEBRTC_AUDIO_ALERTS:
            now = time.perf_counter()
            ALERT_COOLDOWN = 3.0
            if human and now - _WEBRTC_LAST_HUMAN_ALERT > ALERT_COOLDOWN:
                _speak("Human detected")
                _WEBRTC_LAST_HUMAN_ALERT = now
            if gestures:
                gest = gestures[0]["gesture"]
                if now - _WEBRTC_LAST_GESTURE_ALERT > ALERT_COOLDOWN:
                    _speak(f"{gest} detected")
                    _WEBRTC_LAST_GESTURE_ALERT = now
            if faces:
                expr = faces[0]["expression"]
                if now - _WEBRTC_LAST_EXPRESSION_ALERT > ALERT_COOLDOWN:
                    _speak(f"{expr} expression detected")
                    _WEBRTC_LAST_EXPRESSION_ALERT = now

        # ── Save frame ──────────────────────────────────
        _WEBRTC_FRAME_N += 1
        _WEBRTC_TOTAL_DETS += len(dets)
        if _WEBRTC_SAVE and len(dets) > 0 and _WEBRTC_FRAME_N % _WEBRTC_SAVE_INTERVAL == 0:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(_WEBRTC_SAVE_FOLDER, f"frame_{timestamp}_{_WEBRTC_FRAME_N}.jpg")
            cv2.imwrite(filename, to_cpu_mat(ann))
            try:
                existing = sorted([f for f in os.listdir(_WEBRTC_SAVE_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
                while len(existing) > _WEBRTC_MAX_SAVED:
                    os.remove(os.path.join(_WEBRTC_SAVE_FOLDER, existing.pop(0)))
            except Exception:
                pass

        # ── FPS overlay ──────────────────────────────────
        now = time.perf_counter()
        elapsed = now - _WEBRTC_T0
        _WEBRTC_FPS = _WEBRTC_FRAME_N / max(elapsed, 0.001)

        if _WEBRTC_SHOW_FPS:
            cv2.putText(ann, f"FPS  {_WEBRTC_FPS:.1f}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 212, 126), 2, cv2.LINE_AA)
            cv2.putText(ann, _WEBRTC_MODEL_NAME,
                (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                (140, 150, 165), 1, cv2.LINE_AA)

        ts = time.strftime("%H:%M:%S")
        fw_px = ann.shape[1]
        (tsw, _), _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(ann, ts, (fw_px - tsw - 10, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 145), 1, cv2.LINE_AA)

        if _WEBRTC_SHOW_FPS:
            y_off = 68
            if gestures:
                cv2.putText(ann, f"Hands {len(gestures)}",
                    (12, y_off), cv2.FONT_HERSHEY_SIMPLEX,
                    0.36, (34, 211, 238), 1, cv2.LINE_AA)
                y_off += 20
            if faces:
                cv2.putText(ann, f"Faces {len(faces)}",
                    (12, y_off), cv2.FONT_HERSHEY_SIMPLEX,
                    0.36, (200, 130, 255), 1, cv2.LINE_AA)
                y_off += 20
            if _WEBRTC_PSYS is not None and _WEBRTC_PSYS.count > 0:
                cv2.putText(ann, f"Particles {_WEBRTC_PSYS.count}",
                    (12, y_off), cv2.FONT_HERSHEY_SIMPLEX,
                    0.36, (200, 200, 100), 1, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(to_cpu_mat(ann), format="bgr24")

    # ── Side panel display (auto-refresh) ───────────────
    def _render_panel():
        fps_ph.markdown(f"""
        <div class="sp">
            <div class="sp-head">
                <div class="sp-icon green">◈</div>
                <span class="sp-title">Performance</span>
            </div>
            <div class="fps-big">
                <span class="fps-num">{_WEBRTC_FPS:.1f}</span>
                <span class="fps-unit">fps</span>
            </div>
            <div class="fps-sub">Frame #{_WEBRTC_FRAME_N:,}</div>
        </div>""", unsafe_allow_html=True)

        if _WEBRTC_HUMAN:
            alert_ph.markdown("""
            <div class="alert-live">
                <div class="adot"></div>
                <span class="atxt">Human Detected</span>
            </div>""", unsafe_allow_html=True)
        else:
            alert_ph.markdown("""
            <div class="alert-clear">
                <div class="adot"></div>
                <span class="atxt">Scene Clear</span>
            </div>""", unsafe_allow_html=True)

        det_counts = _WEBRTC_DET_COUNTS or collections.Counter()
        if det_counts:
            rows_html = ""
            for cls, cnt in det_counts.most_common(8):
                mk = "person" if cls == "person" else "obj"
                rows_html += f"""
                <div class="det-row">
                    <div class="det-left">
                        <div class="det-marker {mk}"></div>
                        <span class="det-label">{cls}</span>
                    </div>
                    <span class="det-badge">{cnt}</span>
                </div>"""
            detlist_ph.markdown(f"""
            <div class="sp">
                <div class="sp-head">
                    <div class="sp-icon blue">◎</div>
                    <span class="sp-title">Detections</span>
                </div>{rows_html}
            </div>""", unsafe_allow_html=True)
        else:
            detlist_ph.markdown("""
            <div class="sp">
                <div class="sp-head">
                    <div class="sp-icon blue">◎</div>
                    <span class="sp-title">Detections</span>
                </div>
                <div class="det-empty">
                    <span class="det-icon">◉</span>
                    <span>Scanning scene…</span>
                </div>
            </div>""", unsafe_allow_html=True)

        if _WEBRTC_HAND_DETECT:
            gestures = _WEBRTC_GESTURES
            if gestures:
                gesture_rows = ""
                for g in gestures:
                    gesture_rows += f"""
                    <div class="det-row">
                        <div class="det-left">
                            <div class="det-marker" style="background:var(--cyan);"></div>
                            <span class="det-label">{g['gesture']}</span>
                        </div>
                        <span class="det-badge" style="background:var(--cyan-dim);color:var(--cyan);">{g['handedness']}</span>
                    </div>"""
                gesture_ph.markdown(f"""
                <div class="sp">
                    <div class="sp-head">
                        <div class="sp-icon" style="background:var(--cyan-dim);color:var(--cyan);">✋</div>
                        <span class="sp-title">Hand Gestures</span>
                    </div>{gesture_rows}
                </div>""", unsafe_allow_html=True)
            else:
                gesture_ph.markdown("""
                <div class="sp">
                    <div class="sp-head">
                        <div class="sp-icon" style="background:var(--cyan-dim);color:var(--cyan);">✋</div>
                        <span class="sp-title">Hand Gestures</span>
                    </div>
                    <div class="det-empty">
                        <span class="det-icon">✋</span>
                        <span>No hands detected</span>
                    </div>
                </div>""", unsafe_allow_html=True)

        if _WEBRTC_FACE_DETECT:
            faces = _WEBRTC_FACES
            if faces:
                expr_rows = ""
                for f in faces:
                    expr_rows += f"""
                    <div class="det-row">
                        <div class="det-left">
                            <div class="det-marker" style="background:rgb(200,130,255);"></div>
                            <span class="det-label">{f['expression']}</span>
                        </div>
                    </div>"""
                expr_ph.markdown(f"""
                <div class="sp">
                    <div class="sp-head">
                        <div class="sp-icon" style="background:rgba(200,130,255,0.10);color:rgb(200,130,255);">☺</div>
                        <span class="sp-title">Expressions</span>
                    </div>{expr_rows}
                </div>""", unsafe_allow_html=True)
            else:
                expr_ph.markdown("""
                <div class="sp">
                    <div class="sp-head">
                        <div class="sp-icon" style="background:rgba(200,130,255,0.10);color:rgb(200,130,255);">☺</div>
                        <span class="sp-title">Expressions</span>
                    </div>
                    <div class="det-empty">
                        <span class="det-icon">☺</span>
                        <span>No faces detected</span>
                    </div>
                </div>""", unsafe_allow_html=True)

        elapsed = time.perf_counter() - _WEBRTC_T0
        m, s = divmod(int(elapsed), 60)
        avg = _WEBRTC_TOTAL_DETS / max(_WEBRTC_FRAME_N, 1)
        session_ph.markdown(f"""
        <div class="sp">
            <div class="sp-head">
                <div class="sp-icon violet">▦</div>
                <span class="sp-title">Session</span>
            </div>
            <div class="srow">
                <span class="srow-label">Total detections</span>
                <span class="srow-value">{_WEBRTC_TOTAL_DETS:,}</span>
            </div>
            <div class="srow">
                <span class="srow-label">Runtime</span>
                <span class="srow-value">{m:02d}:{s:02d}</span>
            </div>
            <div class="srow">
                <span class="srow-label">Unique classes</span>
                <span class="srow-value">{len(det_counts)}</span>
            </div>
            <div class="srow">
                <span class="srow-label">Avg / frame</span>
                <span class="srow-value">{avg:.1f}</span>
            </div>
        </div>""", unsafe_allow_html=True)

    _render_panel()

    with col_feed:
        webrtc_streamer(
            key="object-detection",
            video_frame_callback=video_frame_callback,
            async_processing=True,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 1280, "max": 1920},
                    "height": {"ideal": 720, "max": 1080},
                    "frameRate": {"ideal": 30, "max": 60},
                },
                "audio": False,
            },
            desired_playing_state=True,
        )

    # ── Saved Frames (below feed, same as local camera) ──
    save_dir = config.get("save_folder", "./saved_frames")
    saved_count = 0
    if os.path.isdir(save_dir):
        saved_count = len([f for f in os.listdir(save_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    if saved_count > 0:
        st.markdown('<hr class="hdivider"/>', unsafe_allow_html=True)
        frames = sorted(
            [f for f in os.listdir(save_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))],
            reverse=True,
        )
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:16px;">
            <span style="font-size:1.1rem;">⬛</span>
            <span style="font-size:0.85rem;font-weight:700;color:var(--text-primary);">Saved Frames</span>
            <span style="font-size:0.70rem;color:var(--text-tertiary);">{len(frames)} captured</span>
        </div>""", unsafe_allow_html=True)

        # Clear all button
        clr_c1, clr_c2 = st.columns([1, 5])
        with clr_c1:
            if st.button("Clear All", key="clear_frames_webrtc"):
                for fname in frames:
                    try:
                        os.remove(os.path.join(save_dir, fname))
                    except Exception:
                        pass
                st.session_state["gallery_page"] = 1
                st.rerun()

        # Pagination
        per_page = 6
        total_pages = max(1, (len(frames) + per_page - 1) // per_page)
        if "gallery_page" not in st.session_state:
            st.session_state["gallery_page"] = 1
        page = st.session_state["gallery_page"]
        page = max(1, min(page, total_pages))
        start = (page - 1) * per_page
        end = start + per_page

        cols = st.columns(min(len(frames[start:end]), 3))
        for i, fname in enumerate(frames[start:end]):
            with cols[i % 3]:
                fpath = os.path.join(save_dir, fname)
                img = Image.open(fpath)
                st.image(img, use_container_width=True)
                with open(fpath, "rb") as f:
                    st.download_button(
                        "▼ Download",
                        data=f.read(),
                        file_name=fname,
                        mime="image/jpeg",
                        key=f"dl_w_{fname}",
                    )

        # Page navigation
        nav_c1, nav_c2, nav_c3 = st.columns([1, 2, 1])
        with nav_c2:
            st.markdown(
                f'<div style="text-align:center;font-size:0.75rem;color:var(--text-tertiary);">'
                f'Page {page} of {total_pages}</div>',
                unsafe_allow_html=True)
        with nav_c1:
            if page > 1 and st.button("◄ Prev", key="gal_prev_w"):
                st.session_state["gallery_page"] = page - 1
                st.rerun()
        with nav_c3:
            if page < total_pages and st.button("Next ►", key="gal_next_w"):
                st.session_state["gallery_page"] = page + 1
                st.rerun()


def run_detection(config, model):
    # Stop button — top of page, same position as Start button
    stop_col, _ = st.columns([1, 3])
    with stop_col:
        stop = st.button("⏹ Stop Detection", type="secondary", key="stop_btn")
    if stop:
        st.session_state["running"] = False
        st.rerun()
        return

    col_feed, col_panel = st.columns([3, 1], gap="large")

    with col_feed:
        st.markdown("""
        <div class="action-bar">
            <div class="action-bar-left">
                <div class="action-bar-title">Live Detection Feed</div>
                <div class="action-bar-sub">Real-time inference — adjust settings in the sidebar</div>
            </div>
        </div>""", unsafe_allow_html=True)
        feed_ph = st.empty()

    with col_panel:
        st.markdown('<div style="height:48px;"></div>', unsafe_allow_html=True)
        fps_ph     = st.empty()
        alert_ph   = st.empty()
        detlist_ph = st.empty()
        gesture_ph = st.empty()
        expr_ph    = st.empty()
        session_ph = st.empty()

    # ── Camera init ────────────────────────────────────
    cap = cv2.VideoCapture(config["camera_index"], cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(config["camera_index"])
    if not cap.isOpened():
        st.error(f"Camera {config['camera_index']} unavailable.")
        st.session_state["running"] = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config["resolution"][0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["resolution"][1])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, config["max_fps"])

    hand_detector = None
    cached_gestures = []
    HAND_SKIP = 3
    if config["hand_detect"]:
        hand_detector = _get_hand_detector(min_conf=config["hand_conf"])

    face_detector = None
    cached_faces = []
    FACE_SKIP = 4
    if config["face_detect"]:
        face_detector = _get_face_detector(min_conf=config["face_conf"])

    psys = None
    if config["particle_fx"] and config["hand_detect"]:
        psys = ParticleSystem(max_particles=config["particle_n"])
    prev_t = time.perf_counter()

    fps_ctr    = FPSCounter(30)
    frame_n    = 0
    total_dets = 0
    t_start    = time.perf_counter()
    min_ft     = 1.0 / config["max_fps"]
    failures   = 0
    MAX_FAIL   = 10

    # ── Audio alert debouncing ───────────────────────────
    if config["audio_alerts"]:
        _init_tts()
        _speak("Audio alerts enabled")
    last_human_alert = 0
    last_gesture_time = 0
    last_expression_time = 0
    ALERT_COOLDOWN = 3.0

    # ── Create save folder if needed ─────────────────────
    if config["save_frames"]:
        os.makedirs(config["save_folder"], exist_ok=True)

    # ── Main loop ──────────────────────────────────────
    # FIX 2 cont: We check st.session_state["running"] each iteration.
    # If the stop button fired (above), running is already False and we
    # exit gracefully. The cap.release() in finally is still safe.
    try:
        while st.session_state.get("running", False):
            t0 = time.perf_counter()

            ret, raw = cap.read()
            if not ret or raw is None:
                failures += 1
                if failures >= MAX_FAIL:
                    st.warning("Camera stopped responding.")
                    break
                time.sleep(0.05)
                continue
            failures = 0

            frame = to_cpu_mat(raw)
            if config["flip_camera"]:
                frame = to_cpu_mat(cv2.flip(frame, 1))

            try:
                ann, dets, det_counts, human = detect_objects(
                    frame, model, config["confidence"], config["iou"])
            except Exception as exc:
                st.warning(f"Inference error: {exc}")
                ann, dets = frame, []
                det_counts, human = collections.Counter(), False

            if hand_detector is not None and frame_n % HAND_SKIP == 0:
                try:
                    small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                    cached_gestures = detect_hands(small, hand_detector)
                except Exception:
                    cached_gestures = []
            gestures = cached_gestures if hand_detector is not None else []
            if gestures:
                ann = draw_hands(ann, gestures)

            if face_detector is not None and frame_n % FACE_SKIP == 0:
                try:
                    small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                    cached_faces = detect_faces(small, face_detector)
                except Exception:
                    cached_faces = []
            faces = cached_faces if face_detector is not None else []
            if faces:
                ann = draw_faces(ann, faces)

            if psys is not None:
                now_t = time.perf_counter()
                dt    = min(now_t - prev_t, 0.1)
                prev_t = now_t
                fh, fw = ann.shape[:2]
                if gestures:
                    g   = gestures[0]
                    lm  = g["landmarks"].landmark
                    pcx, pcy = _palm_center(lm, fw, fh)
                    gest = g["gesture"]
                    dx, dy = None, None
                    if gest in ("Pointing", "Peace"):
                        dx, dy = _pointing_dir(lm, fw, fh)
                    _spawn_for_gesture(psys, gest, pcx, pcy, dx, dy)
                    psys.update(dt, gest, pcx, pcy, dx, dy, fw, fh)
                else:
                    psys.update(dt, fw=fw, fh=fh)
                ann = psys.render(ann)

            # ── Audio alerts ─────────────────────────────────────
            if config["audio_alerts"]:
                now = time.perf_counter()
                if human and now - last_human_alert > ALERT_COOLDOWN:
                    _speak("Human detected")
                    last_human_alert = now
                if gestures:
                    gest = gestures[0]["gesture"]
                    if now - last_gesture_time > ALERT_COOLDOWN:
                        print(f"[DEBUG] Gesture alert: {gest}")
                        _speak(f"{gest} detected")
                        last_gesture_time = now
                if faces:
                    expr = faces[0]["expression"]
                    if now - last_expression_time > ALERT_COOLDOWN:
                        print(f"[DEBUG] Expression alert: {expr}")
                        _speak(f"{expr} expression detected")
                        last_expression_time = now

            # ── Save frame if enabled, detections exist, and interval matches ───────
            if config["save_frames"] and len(dets) > 0 and frame_n % config["save_interval"] == 0:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"{config['save_folder']}/frame_{timestamp}_{frame_n}.jpg"
                cv2.imwrite(filename, ann)
                # Auto-cleanup: delete oldest frames if over max
                try:
                    existing = sorted([f for f in os.listdir(config["save_folder"]) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
                    while len(existing) > config["max_saved"]:
                        os.remove(os.path.join(config["save_folder"], existing.pop(0)))
                except Exception:
                    pass

            fps_ctr.tick()
            frame_n    += 1
            total_dets += len(dets)
            cur_fps     = fps_ctr.fps()

            if config["show_fps"]:
                cv2.putText(ann, f"FPS  {cur_fps:.1f}",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 212, 126), 2, cv2.LINE_AA)
                cv2.putText(ann, config["model"].upper(),
                    (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    (140, 150, 165), 1, cv2.LINE_AA)

            ts = time.strftime("%H:%M:%S")
            fw_px = ann.shape[1]
            (tsw, _), _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.putText(ann, ts, (fw_px - tsw - 10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 145), 1, cv2.LINE_AA)

            if config["show_fps"]:
                y_off = 68
                if gestures:
                    cv2.putText(ann, f"Hands {len(gestures)}",
                        (12, y_off), cv2.FONT_HERSHEY_SIMPLEX,
                        0.36, (34, 211, 238), 1, cv2.LINE_AA)
                    y_off += 20
                if faces:
                    cv2.putText(ann, f"Faces {len(faces)}",
                        (12, y_off), cv2.FONT_HERSHEY_SIMPLEX,
                        0.36, (200, 130, 255), 1, cv2.LINE_AA)
                    y_off += 20
                if psys is not None and psys.count > 0:
                    cv2.putText(ann, f"Particles {psys.count}",
                        (12, y_off), cv2.FONT_HERSHEY_SIMPLEX,
                        0.36, (200, 200, 100), 1, cv2.LINE_AA)

            _render_image(feed_ph, frame_to_pil(ann), "")

            fps_ph.markdown(f"""
            <div class="sp">
                <div class="sp-head">
                    <div class="sp-icon green">◈</div>
                    <span class="sp-title">Performance</span>
                </div>
                <div class="fps-big">
                    <span class="fps-num">{cur_fps:.1f}</span>
                    <span class="fps-unit">fps</span>
                </div>
                <div class="fps-sub">Frame #{frame_n:,}</div>
            </div>""", unsafe_allow_html=True)

            if human:
                alert_ph.markdown("""
                <div class="alert-live">
                    <div class="adot"></div>
                    <span class="atxt">Human Detected</span>
                </div>""", unsafe_allow_html=True)
            else:
                alert_ph.markdown("""
                <div class="alert-clear">
                    <div class="adot"></div>
                    <span class="atxt">Scene Clear</span>
                </div>""", unsafe_allow_html=True)

            if det_counts:
                rows_html = ""
                for cls, cnt in det_counts.most_common(8):
                    mk = "person" if cls == "person" else "obj"
                    rows_html += f"""
                    <div class="det-row">
                        <div class="det-left">
                            <div class="det-marker {mk}"></div>
                            <span class="det-label">{cls}</span>
                        </div>
                        <span class="det-badge">{cnt}</span>
                    </div>"""
                detlist_ph.markdown(f"""
                <div class="sp">
                    <div class="sp-head">
                        <div class="sp-icon blue">◎</div>
                        <span class="sp-title">Detections</span>
                    </div>{rows_html}
                </div>""", unsafe_allow_html=True)
            else:
                detlist_ph.markdown("""
                <div class="sp">
                    <div class="sp-head">
                        <div class="sp-icon blue">◎</div>
                        <span class="sp-title">Detections</span>
                    </div>
                    <div class="det-empty">
                        <span class="det-icon">◉</span>
                        <span>Scanning scene…</span>
                    </div>
                </div>""", unsafe_allow_html=True)

            if config["hand_detect"]:
                if gestures:
                    gesture_rows = ""
                    for g in gestures:
                        gesture_rows += f"""
                        <div class="det-row">
                            <div class="det-left">
                                <div class="det-marker" style="background:var(--cyan);"></div>
                                <span class="det-label">{g['gesture']}</span>
                            </div>
                            <span class="det-badge" style="background:var(--cyan-dim);color:var(--cyan);">{g['handedness']}</span>
                        </div>"""
                    gesture_ph.markdown(f"""
                    <div class="sp">
                        <div class="sp-head">
                            <div class="sp-icon" style="background:var(--cyan-dim);color:var(--cyan);">✋</div>
                            <span class="sp-title">Hand Gestures</span>
                        </div>{gesture_rows}
                    </div>""", unsafe_allow_html=True)
                else:
                    gesture_ph.markdown("""
                    <div class="sp">
                        <div class="sp-head">
                            <div class="sp-icon" style="background:var(--cyan-dim);color:var(--cyan);">✋</div>
                            <span class="sp-title">Hand Gestures</span>
                        </div>
                        <div class="det-empty">
                            <span class="det-icon">✋</span>
                            <span>No hands detected</span>
                        </div>
                    </div>""", unsafe_allow_html=True)

            if config["face_detect"]:
                if faces:
                    expr_rows = ""
                    for f in faces:
                        expr_rows += f"""
                        <div class="det-row">
                            <div class="det-left">
                                <div class="det-marker" style="background:rgb(200,130,255);"></div>
                                <span class="det-label">{f['expression']}</span>
                            </div>
                        </div>"""
                    expr_ph.markdown(f"""
                    <div class="sp">
                        <div class="sp-head">
                            <div class="sp-icon" style="background:rgba(200,130,255,0.10);color:rgb(200,130,255);">☺</div>
                            <span class="sp-title">Expressions</span>
                        </div>{expr_rows}
                    </div>""", unsafe_allow_html=True)
                else:
                    expr_ph.markdown("""
                    <div class="sp">
                        <div class="sp-head">
                            <div class="sp-icon" style="background:rgba(200,130,255,0.10);color:rgb(200,130,255);">☺</div>
                            <span class="sp-title">Expressions</span>
                        </div>
                        <div class="det-empty">
                            <span class="det-icon">☺</span>
                            <span>No faces detected</span>
                        </div>
                    </div>""", unsafe_allow_html=True)

            elapsed = time.perf_counter() - t_start
            m, s = divmod(int(elapsed), 60)
            avg  = total_dets / max(frame_n, 1)
            session_ph.markdown(f"""
            <div class="sp">
                <div class="sp-head">
                    <div class="sp-icon violet">▦</div>
                    <span class="sp-title">Session</span>
                </div>
                <div class="srow">
                    <span class="srow-label">Total detections</span>
                    <span class="srow-value">{total_dets:,}</span>
                </div>
                <div class="srow">
                    <span class="srow-label">Runtime</span>
                    <span class="srow-value">{m:02d}:{s:02d}</span>
                </div>
                <div class="srow">
                    <span class="srow-label">Unique classes</span>
                    <span class="srow-value">{len(det_counts)}</span>
                </div>
                <div class="srow">
                    <span class="srow-label">Avg / frame</span>
                    <span class="srow-value">{avg:.1f}</span>
                </div>
            </div>""", unsafe_allow_html=True)

            sleep_t = max(0.0, min_ft - (time.perf_counter() - t0))
            if sleep_t > 0:
                time.sleep(sleep_t)

    except Exception as exc:
        st.error(f"Detection error: {exc}")
    finally:
        cap.release()
        # FIX 2: Set running=False cleanly. Do NOT call st.rerun() here —
        # rerun() inside a finally block restarts the entire Streamlit
        # script execution which drops the server WebSocket connection
        # and causes the "server stopped" error users were seeing.
        st.session_state["running"] = False



def main():
    if "running" not in st.session_state:
        st.session_state["running"] = False

    config = render_sidebar()

    loader_ph = st.empty()
    with loader_ph.container():
        with st.spinner(f"Loading {config['model']}…"):
            model = load_yolo_model(config["model"])

    if model is None:
        st.error("Failed to load model.")
        st.code("pip install ultralytics torch torchvision", language="bash")
        return
    loader_ph.empty()

    device    = getattr(model, "_infer_device", "cpu")
    dev_label = "CUDA" if device == "cuda" else "CPU"
    dev_cls   = "green" if device == "cuda" else "amber"
    is_live   = st.session_state["running"]

    # Status dot class
    status_dot = "live" if is_live else ""
    status_txt = "Live" if is_live else "Idle"

    st.markdown(f"""
    <div class="topbar">
        <div class="topbar-brand">
            <div class="brand-mark">◉</div>
            <div class="brand-text">
                <div class="brand-name">Live Object Detection & <span>Tracing</span></div>
                <div class="brand-tagline">Real-time object detection — YOLOv8</div>
            </div>
        </div>
        <div class="topbar-status">
            <div class="status-dot {status_dot}"></div>
            {status_txt}
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-strip">
        <div class="metric-tile">
            <div class="metric-tile-header">
                <span class="metric-tile-label">Model</span>
                <div class="metric-tile-icon blue">⬡</div>
            </div>
            <div class="metric-tile-value">{config['model'].upper()}</div>
            <div class="metric-tile-sub">YOLOv8 variant</div>
        </div>
        <div class="metric-tile">
            <div class="metric-tile-header">
                <span class="metric-tile-label">Compute</span>
                <div class="metric-tile-icon {dev_cls}">⚡</div>
            </div>
            <div class="metric-tile-value {dev_cls}">{dev_label}</div>
            <div class="metric-tile-sub">Inference device</div>
        </div>
        <div class="metric-tile">
            <div class="metric-tile-header">
                <span class="metric-tile-label">Classes</span>
                <div class="metric-tile-icon violet">⬛</div>
            </div>
            <div class="metric-tile-value">{len(model.names)}</div>
            <div class="metric-tile-sub">Detectable objects</div>
        </div>
        <div class="metric-tile">
            <div class="metric-tile-header">
                <span class="metric-tile-label">Camera</span>
                <div class="metric-tile-icon amber">◎</div>
            </div>
            <div class="metric-tile-value">ID&nbsp;{config['camera_index']}</div>
            <div class="metric-tile-sub">{config['resolution'][0]} × {config['resolution'][1]}</div>
        </div>
    </div>""", unsafe_allow_html=True)

    with st.expander("Detectable classes", expanded=False):
        tags = ""
        for cls in model.names.values():
            t_cls = "person" if cls == "person" else ""
            tags += f'<span class="class-pill {t_cls}">{cls}</span>'
        st.markdown(f'<div style="padding:10px 4px 4px;line-height:2.4;">{tags}</div>', unsafe_allow_html=True)

    st.markdown('<hr class="hdivider"/>', unsafe_allow_html=True)

    if is_live:
        if config["cam_source"] == "Browser Webcam":
            run_browser_detection(config, model)
        else:
            run_detection(config, model)
    else:
        # ── Tab Navigation ──────────────────────────────
        save_dir = config.get("save_folder", "./saved_frames")
        saved_count = 0
        if os.path.isdir(save_dir):
            saved_count = len([f for f in os.listdir(save_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

        tab_detect, tab_gallery = st.tabs([
            f"◉ Detection",
            f"⬛ Saved Frames" + (f" ({saved_count})" if saved_count else ""),
        ])

        # ── Detection Tab ───────────────────────────────
        with tab_detect:
            btn_c, _ = st.columns([1, 3])
            with btn_c:
                if st.button("► Start Detection", type="primary", key="start_btn"):
                    st.session_state["running"] = True
                    st.rerun()

            st.markdown("""
            <div class="hero-wrap">
                <div class="hero-icon-ring">◎</div>
                <div class="hero-heading">Ready to Detect</div>
                <div class="hero-body">
                    Configure your model and camera in the sidebar,
                    then press <strong style="color:var(--green);">Start Detection</strong>
                    to begin real-time inference.
                </div>
                <div class="hero-chips">
                    <div class="h-chip"><span style="color:var(--green);">◈</span> Real-time inference</div>
                    <div class="h-chip"><span style="color:var(--blue);">◎</span> 80+ object classes</div>
                    <div class="h-chip"><span style="color:var(--red);">⬡</span> Human detection alert</div>
                    <div class="h-chip"><span style="color:var(--amber);">▦</span> Live session analytics</div>
                    <div class="h-chip"><span style="color:var(--cyan);">✋</span> Hand gesture detection</div>
                    <div class="h-chip"><span style="color:rgb(200,130,255);">☺</span> Face expression detection</div>
                    <div class="h-chip"><span style="color:rgb(200,200,100);">✦</span> Gesture particle effects</div>
                </div>
            </div>""", unsafe_allow_html=True)

        # ── Saved Frames Tab ────────────────────────────
        with tab_gallery:
            if not os.path.isdir(save_dir) or saved_count == 0:
                st.markdown("""
                <div style="text-align:center;padding:60px 20px;">
                    <div style="font-size:2.5rem;opacity:0.3;margin-bottom:12px;">⬛</div>
                    <div style="font-size:0.95rem;font-weight:600;color:var(--text-secondary);margin-bottom:6px;">No Saved Frames Yet</div>
                    <div style="font-size:0.78rem;color:var(--text-tertiary);max-width:340px;margin:0 auto;">
                        Enable <strong>Save Frames</strong> in the sidebar and run detection to capture frames.
                        They will appear here for preview and download.
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                frames = sorted(
                    [f for f in os.listdir(save_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))],
                    reverse=True,
                )
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:16px;">
                    <span style="font-size:1.1rem;">⬛</span>
                    <span style="font-size:0.85rem;font-weight:700;color:var(--text-primary);">Saved Frames</span>
                    <span style="font-size:0.70rem;color:var(--text-tertiary);">{len(frames)} captured</span>
                </div>""", unsafe_allow_html=True)

                # Clear all button
                clr_c1, clr_c2 = st.columns([1, 5])
                with clr_c1:
                    if st.button("Clear All", key="clear_frames"):
                        for fname in frames:
                            try:
                                os.remove(os.path.join(save_dir, fname))
                            except Exception:
                                pass
                        st.session_state["gallery_page"] = 1
                        st.rerun()

                # Pagination
                per_page = 6
                total_pages = max(1, (len(frames) + per_page - 1) // per_page)
                if "gallery_page" not in st.session_state:
                    st.session_state["gallery_page"] = 1
                page = st.session_state["gallery_page"]
                page = max(1, min(page, total_pages))
                start = (page - 1) * per_page
                page_frames = frames[start:start + per_page]

                cols = st.columns(min(len(page_frames), 3))
                for i, fname in enumerate(page_frames):
                    with cols[i % 3]:
                        fpath = os.path.join(save_dir, fname)
                        img = Image.open(fpath)
                        st.image(img, use_container_width=True)
                        with open(fpath, "rb") as f:
                            st.download_button(
                                "▼ Download",
                                data=f.read(),
                                file_name=fname,
                                mime="image/jpeg",
                                key=f"dl_{fname}",
                            )

                # Pagination controls
                nav_c1, nav_c2, nav_c3 = st.columns([1, 2, 1])
                with nav_c1:
                    if page > 1 and st.button("◄ Prev", key="gal_prev"):
                        st.session_state["gallery_page"] = page - 1
                        st.rerun()
                with nav_c3:
                    if page < total_pages and st.button("Next ►", key="gal_next"):
                        st.session_state["gallery_page"] = page + 1
                        st.rerun()
                with nav_c2:
                    st.markdown(
                        f'<div style="text-align:center;font-size:0.75rem;color:var(--text-tertiary);">'
                        f'Page {page} of {total_pages}</div>',
                        unsafe_allow_html=True,
                    )


if __name__ == "__main__":
    main()