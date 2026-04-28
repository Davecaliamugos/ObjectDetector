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
    page_title="VisionAI — Object Detection",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<link href="https://cdn.jsdelivr.net/npm/remixicon@4.1.0/fonts/remixicon.css" rel="stylesheet">

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
   GLOBAL BASE
═══════════════════════════════════════════════════════════ */
*, *::before, *::after {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    box-sizing: border-box;
}
code, .mono, [class*="mono"] {
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
}

html, .stApp {
    background: var(--bg-base) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: var(--bg-elevated);
    border-radius: 99px;
}
::-webkit-scrollbar-thumb:hover { background: var(--bg-hover); }

/* Main content padding */
.block-container {
    padding: 0 2.5rem 3rem !important;
    max-width: 1400px !important;
}

/* ═══════════════════════════════════════════════════════════
   TOPBAR / HEADER
═══════════════════════════════════════════════════════════ */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 28px 0 20px;
    border-bottom: 1px solid var(--border-faint);
    margin-bottom: 28px;
}
.topbar-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}
.brand-mark {
    width: 38px;
    height: 38px;
    background: linear-gradient(145deg, #00d47e 0%, #4f8ef7 100%);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 0 24px rgba(0,212,126,0.20);
    flex-shrink: 0;
}
.brand-mark i {
    font-size: 18px;
    color: #fff;
}
.brand-text {
    display: flex;
    flex-direction: column;
    gap: 1px;
}
.brand-name {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    line-height: 1;
}
.brand-name span { color: var(--green); }
.brand-tagline {
    font-size: 0.68rem;
    font-weight: 400;
    color: var(--text-tertiary);
    letter-spacing: 0.03em;
}
.topbar-status {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 99px;
    font-size: 0.72rem;
    font-weight: 500;
    color: var(--text-secondary);
}
.status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--green);
}
.status-dot.idle { background: var(--text-muted); }
.status-dot.live  {
    background: var(--red);
    animation: pulse-dot 1.4s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:0.5; transform:scale(0.75); }
}

/* ═══════════════════════════════════════════════════════════
   METRIC STRIP
═══════════════════════════════════════════════════════════ */
.metric-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 24px;
}
.metric-tile {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--r-lg);
    padding: 18px 20px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    transition: border-color var(--dur) var(--ease),
                transform var(--dur) var(--ease);
    cursor: default;
}
.metric-tile:hover {
    border-color: var(--border-default);
    transform: translateY(-2px);
}
.metric-tile-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.metric-tile-label {
    font-size: 0.67rem;
    font-weight: 600;
    color: var(--text-tertiary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.metric-tile-icon {
    width: 24px;
    height: 24px;
    border-radius: var(--r-sm);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
}
.metric-tile-icon.green { background:var(--green-dim); color:var(--green); }
.metric-tile-icon.blue  { background:var(--blue-dim);  color:var(--blue);  }
.metric-tile-icon.amber { background:var(--amber-dim); color:var(--amber); }
.metric-tile-icon.red   { background:var(--red-dim);   color:var(--red);   }
.metric-tile-icon.violet{ background:var(--violet-dim);color:var(--violet);}

.metric-tile-value {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.03em;
    font-family: 'JetBrains Mono', monospace !important;
    line-height: 1;
}
.metric-tile-value.green  { color: var(--green);  }
.metric-tile-value.blue   { color: var(--blue);   }
.metric-tile-value.amber  { color: var(--amber);  }
.metric-tile-value.red    { color: var(--red);    }
.metric-tile-sub {
    font-size: 0.68rem;
    color: var(--text-muted);
    margin-top: -4px;
}

/* ═══════════════════════════════════════════════════════════
   DIVIDER
═══════════════════════════════════════════════════════════ */
.hdivider {
    height: 1px;
    background: linear-gradient(90deg,
        transparent 0%,
        var(--border-subtle) 30%,
        var(--border-subtle) 70%,
        transparent 100%);
    margin: 20px 0;
    border: none;
}

/* ═══════════════════════════════════════════════════════════
   CLASSES PANEL (expander override)
═══════════════════════════════════════════════════════════ */
details {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--r-lg) !important;
    overflow: hidden;
    margin-bottom: 20px !important;
}
details summary {
    padding: 14px 18px !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    cursor: pointer;
    user-select: none;
    letter-spacing: 0.01em;
}
details summary:hover { color: var(--text-primary) !important; }
details[open] summary {
    border-bottom: 1px solid var(--border-faint);
}

.class-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: var(--bg-elevated);
    border: 1px solid var(--border-faint);
    padding: 3px 10px;
    border-radius: var(--r-xs);
    font-size: 0.70rem;
    font-weight: 500;
    color: var(--text-secondary);
    font-family: 'JetBrains Mono', monospace !important;
    margin: 2px;
    transition: border-color var(--dur) var(--ease);
}
.class-pill:hover { border-color: var(--border-default); }
.class-pill.person {
    border-color: var(--red-border);
    color: var(--red);
    background: var(--red-dim);
}

/* ═══════════════════════════════════════════════════════════
   ACTION BAR  (button row above feed)
═══════════════════════════════════════════════════════════ */
.action-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
}
.action-bar-left {
    display: flex;
    flex-direction: column;
    gap: 2px;
}
.action-bar-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: -0.01em;
}
.action-bar-sub {
    font-size: 0.72rem;
    color: var(--text-tertiary);
}

/* ═══════════════════════════════════════════════════════════
   STREAMLIT BUTTON OVERRIDES
═══════════════════════════════════════════════════════════ */
div[data-testid="stButton"] > button {
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.01em !important;
    border-radius: var(--r-md) !important;
    height: 42px !important;
    border: none !important;
    transition: all var(--dur) var(--ease) !important;
    white-space: nowrap !important;
}
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, var(--green) 0%, #00b36a 100%) !important;
    color: #05120d !important;
    box-shadow: 0 0 22px var(--green-glow) !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    box-shadow: 0 0 34px var(--green-glow) !important;
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
   VIDEO FRAME WRAPPER
═══════════════════════════════════════════════════════════ */
.feed-wrapper {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--r-xl);
    padding: 8px;
    position: relative;
}
.feed-wrapper img {
    border-radius: calc(var(--r-xl) - 6px);
    display: block;
    width: 100%;
}
.feed-badge-row {
    position: absolute;
    top: 18px;
    left: 18px;
    display: flex;
    gap: 6px;
}
.feed-badge {
    display: flex;
    align-items: center;
    gap: 6px;
    background: rgba(8,8,14,0.75);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 5px 12px;
    border-radius: 99px;
    font-size: 0.68rem;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.feed-badge .live-pulse {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--red);
    animation: pulse-dot 1.4s ease-in-out infinite;
}

/* ═══════════════════════════════════════════════════════════
   HERO / IDLE STATE
═══════════════════════════════════════════════════════════ */
.hero-wrap {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--r-xl);
    padding: 72px 40px 64px;
    text-align: center;
    position: relative;
    overflow: hidden;
    min-height: 420px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.hero-wrap::after {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse 60% 50% at 50% 0%,
        rgba(0,212,126,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-icon-ring {
    width: 80px; height: 80px;
    margin: 0 auto 28px;
    background: var(--bg-elevated);
    border: 1px solid var(--border-default);
    border-radius: 22px;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    z-index: 1;
}
.hero-icon-ring i {
    font-size: 32px;
    background: linear-gradient(135deg, var(--green), var(--blue));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-heading {
    font-size: 1.55rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.03em;
    margin-bottom: 10px;
    position: relative;
    z-index: 1;
}
.hero-body {
    font-size: 0.85rem;
    color: var(--text-tertiary);
    line-height: 1.7;
    max-width: 460px;
    margin: 0 auto 32px;
    position: relative;
    z-index: 1;
    font-weight: 400;
}
.hero-chips {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 8px;
    position: relative;
    z-index: 1;
}
.h-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: var(--r-sm);
    padding: 7px 14px;
    font-size: 0.73rem;
    font-weight: 500;
    color: var(--text-secondary);
    transition: all var(--dur) var(--ease);
}
.h-chip:hover {
    border-color: var(--border-default);
    color: var(--text-primary);
    transform: translateY(-1px);
}
.h-chip i { font-size: 13px; }
.h-chip.green i { color: var(--green); }
.h-chip.blue  i { color: var(--blue);  }
.h-chip.red   i { color: var(--red);   }
.h-chip.amber i { color: var(--amber); }

/* ═══════════════════════════════════════════════════════════
   STATS SIDEBAR PANELS
═══════════════════════════════════════════════════════════ */
.sp {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--r-lg);
    padding: 16px 18px;
    margin-bottom: 10px;
    transition: border-color var(--dur) var(--ease);
}
.sp:hover { border-color: var(--border-default); }

.sp-head {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 14px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border-faint);
}
.sp-icon {
    width: 26px; height: 26px;
    border-radius: var(--r-sm);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    flex-shrink: 0;
}
.sp-icon.green  { background:var(--green-dim);  color:var(--green);  }
.sp-icon.blue   { background:var(--blue-dim);   color:var(--blue);   }
.sp-icon.red    { background:var(--red-dim);    color:var(--red);    }
.sp-icon.violet { background:var(--violet-dim); color:var(--violet); }
.sp-icon.amber  { background:var(--amber-dim);  color:var(--amber);  }

.sp-title {
    font-size: 0.69rem;
    font-weight: 700;
    color: var(--text-tertiary);
    text-transform: uppercase;
    letter-spacing: 0.07em;
}

/* FPS big number */
.fps-big {
    display: flex;
    align-items: baseline;
    gap: 5px;
    margin-bottom: 6px;
}
.fps-num {
    font-size: 2.4rem;
    font-weight: 800;
    color: var(--green);
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: -0.04em;
    line-height: 1;
}
.fps-unit {
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--text-tertiary);
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.fps-sub {
    font-size: 0.68rem;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace !important;
}

/* Alert badges */
.alert-live {
    background: var(--red-dim);
    border: 1px solid var(--red-border);
    border-radius: var(--r-md);
    padding: 11px 14px;
    display: flex;
    align-items: center;
    gap: 10px;
    box-shadow: 0 0 18px var(--red-glow);
    animation: breathe-red 2s ease-in-out infinite;
}
@keyframes breathe-red {
    0%,100% { box-shadow: 0 0 14px var(--red-glow); }
    50%      { box-shadow: 0 0 26px var(--red-glow); }
}
.alert-live .adot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--red);
    animation: pulse-dot 1.2s ease-in-out infinite;
    flex-shrink: 0;
}
.alert-live .atxt {
    font-size: 0.76rem;
    font-weight: 700;
    color: var(--red);
    letter-spacing: 0.04em;
}
.alert-clear {
    background: var(--green-dim);
    border: 1px solid var(--green-border);
    border-radius: var(--r-md);
    padding: 11px 14px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.alert-clear .adot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--green);
    flex-shrink: 0;
}
.alert-clear .atxt {
    font-size: 0.76rem;
    font-weight: 600;
    color: var(--green);
}

/* Detection list */
.det-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 7px 0;
    border-bottom: 1px solid var(--border-faint);
}
.det-row:last-child { border-bottom: none; }
.det-left {
    display: flex;
    align-items: center;
    gap: 8px;
}
.det-marker {
    width: 5px; height: 5px;
    border-radius: 50%;
    flex-shrink: 0;
}
.det-marker.person { background: var(--red); }
.det-marker.obj    { background: var(--blue); }
.det-label {
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--text-primary);
    text-transform: capitalize;
}
.det-badge {
    font-size: 0.68rem;
    font-weight: 700;
    color: var(--text-secondary);
    background: var(--bg-elevated);
    border: 1px solid var(--border-faint);
    padding: 2px 9px;
    border-radius: 99px;
    font-family: 'JetBrains Mono', monospace !important;
}
.det-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 20px 0 8px;
    color: var(--text-muted);
}
.det-empty i { font-size: 22px; opacity: 0.4; }
.det-empty span { font-size: 0.75rem; }

/* Session rows */
.srow {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    border-bottom: 1px solid var(--border-faint);
}
.srow:last-child { border-bottom: none; }
.srow-label {
    font-size: 0.73rem;
    color: var(--text-tertiary);
    font-weight: 400;
}
.srow-value {
    font-size: 0.76rem;
    font-weight: 700;
    color: var(--text-primary);
    font-family: 'JetBrains Mono', monospace !important;
}

/* ═══════════════════════════════════════════════════════════
   ACTIVE BANNER (replaces start button when running)
═══════════════════════════════════════════════════════════ */
.active-banner {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 10px 20px;
    background: var(--green-dim);
    border: 1px solid var(--green-border);
    border-radius: var(--r-md);
    width: 100%;
}
.active-banner .adot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--green);
    animation: pulse-dot 1.6s ease-in-out infinite;
}
.active-banner .atxt {
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--green);
    letter-spacing: 0.03em;
}

/* ═══════════════════════════════════════════════════════════
   SIDEBAR
═══════════════════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border-faint) !important;
}
section[data-testid="stSidebar"] > div {
    padding: 20px 16px !important;
}

.sb-section {
    margin-bottom: 22px;
}
.sb-heading {
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 0.67rem;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border-faint);
}
.sb-heading i { font-size: 12px; }

/* Sidebar system info */
.sysbox {
    background: var(--bg-overlay);
    border: 1px solid var(--border-faint);
    border-radius: var(--r-md);
    padding: 12px 14px;
}
.sysrow {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    border-bottom: 1px solid var(--border-faint);
}
.sysrow:last-child { border-bottom: none; }
.sysrow-lbl {
    font-size: 0.71rem;
    color: var(--text-tertiary);
}
.sysrow-val {
    font-size: 0.71rem;
    font-weight: 600;
    color: var(--text-primary);
    font-family: 'JetBrains Mono', monospace !important;
}
.chip-sm {
    display: inline-flex;
    align-items: center;
    gap: 3px;
    padding: 2px 8px;
    border-radius: var(--r-xs);
    font-size: 0.66rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: 0.04em;
}
.chip-sm.gpu { background:var(--green-dim); color:var(--green); }
.chip-sm.cpu { background:var(--amber-dim); color:var(--amber); }

/* Slider / selectbox / checkbox label polish */
label[data-testid="stWidgetLabel"] p {
    font-size: 0.76rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
}
div[data-testid="stSlider"] { padding-bottom: 4px; }

/* ═══════════════════════════════════════════════════════════
   HIDE STREAMLIT CHROME
═══════════════════════════════════════════════════════════ */
#MainMenu, header, footer,
div[data-testid="stToolbar"],
div[data-testid="stDecoration"],
div[data-testid="stStatusWidget"],
div[data-testid="metric-container"] {
    display: none !important;
    visibility: hidden !important;
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


# ============================================================
# PHASE 5: DETECTION ENGINE
# ============================================================
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

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Corner accents
        cl, ct = 14, 3
        for px, py, sx, sy in [
            (x1, y1, 1, 1), (x2, y1, -1, 1),
            (x1, y2, 1, -1), (x2, y2, -1, -1),
        ]:
            cv2.line(frame, (px, py), (px + sx*cl, py), color, ct)
            cv2.line(frame, (px, py), (px, py + sy*cl), color, ct)

        # Label
        label = f"{name}  {conf:.0%}"
        font  = cv2.FONT_HERSHEY_SIMPLEX
        fs, ft = 0.46, 1
        (tw, th), bl = cv2.getTextSize(label, font, fs, ft)
        ly  = max(y1 - 6, th + 8)
        px2, py2 = 6, 4
        cv2.rectangle(frame,
            (x1, ly - th - py2*2),
            (x1 + tw + px2*2, ly + 2),
            color, -1)
        cv2.putText(frame, label,
            (x1 + px2, ly - py2), font, fs,
            (255, 255, 255), ft, cv2.LINE_AA)

        # Centre crosshair
        cx, cy = (x1+x2)//2, (y1+y2)//2
        cv2.drawMarker(frame, (cx, cy), color,
                       cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA)

        # Distance hint
        if is_person:
            bh = y2 - y1
            if bh > 10:
                cv2.putText(frame,
                    f"~{max(1,int(500/bh))}m",
                    (x1, y2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (180, 200, 220), 1, cv2.LINE_AA)

    # Bottom bar
    bar_h = 34
    ov = frame.copy()
    cv2.rectangle(ov, (0, h - bar_h), (w, h), (8, 8, 14), -1)
    cv2.addWeighted(ov, 0.72, frame, 0.28, 0, frame)

    cv2.putText(frame,
        f"{len(detections)} object{'s' if len(detections)!=1 else ''} detected",
        (12, h - 11),
        cv2.FONT_HERSHEY_SIMPLEX, 0.43,
        (150, 160, 175), 1, cv2.LINE_AA)

    if human_detected:
        alert = "HUMAN DETECTED"
        (aw, _), _ = cv2.getTextSize(alert, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1)
        cv2.putText(frame, alert,
            (w - aw - 12, h - 11),
            cv2.FONT_HERSHEY_SIMPLEX, 0.43,
            (71, 68, 239), 1, cv2.LINE_AA)

    return frame


# ============================================================
# PHASE 6: PIL CONVERSION
# ============================================================
def frame_to_pil(frame):
    frame = to_cpu_mat(frame)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8))


# ============================================================
# PHASE 7: FPS COUNTER
# ============================================================
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


# ============================================================
# PHASE 8: st.image COMPAT HELPER
# ============================================================
def _render_image(placeholder, img, caption=""):
    if "_img_kw" not in st.session_state:
        import inspect
        p = inspect.signature(st.image).parameters
        st.session_state["_img_kw"] = (
            "use_container_width" if "use_container_width" in p else "width"
        )
    kw = st.session_state["_img_kw"]
    if kw == "use_container_width":
        placeholder.image(img, use_container_width=True, caption=caption)
    else:
        placeholder.image(img, width=720, caption=caption)


# ============================================================
# PHASE 9: SIDEBAR
# ============================================================
def render_sidebar():
    # ── Sidebar brand ──────────────────────────────────
    st.sidebar.markdown("""
    <div style="display:flex;align-items:center;gap:10px;
                padding:0 2px 18px;
                border-bottom:1px solid var(--border-faint);
                margin-bottom:18px;">
        <div style="width:32px;height:32px;
                    background:linear-gradient(145deg,#00d47e,#4f8ef7);
                    border-radius:9px;display:flex;
                    align-items:center;justify-content:center;
                    box-shadow:0 0 16px rgba(0,212,126,0.18);flex-shrink:0;">
            <i class="ri-eye-line" style="font-size:15px;color:#fff;"></i>
        </div>
        <div>
            <div style="font-size:0.88rem;font-weight:700;
                        color:var(--text-primary);letter-spacing:-0.02em;">
                Vision<span style="color:var(--green);">AI</span>
            </div>
            <div style="font-size:0.63rem;color:var(--text-muted);
                        letter-spacing:0.04em;">Configuration Panel</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Model ──────────────────────────────────────────
    st.sidebar.markdown("""
    <div class="sb-heading">
        <i class="ri-cpu-line"></i> Model
    </div>""", unsafe_allow_html=True)

    model_map = {
        "Nano  — Fastest":       "yolov8n",
        "Small — Balanced":      "yolov8s",
        "Medium — Accurate":     "yolov8m",
        "Large — High Accuracy": "yolov8l",
        "XLarge — Maximum":      "yolov8x",
    }
    label = st.sidebar.selectbox(
        "Variant", list(model_map.keys()), index=0,
        label_visibility="collapsed",
    )

    # ── Thresholds ─────────────────────────────────────
    st.sidebar.markdown("""
    <div class="sb-heading" style="margin-top:18px;">
        <i class="ri-equalizer-line"></i> Detection Thresholds
    </div>""", unsafe_allow_html=True)

    conf = st.sidebar.slider(
        "Confidence threshold", 0.10, 1.0, 0.50, 0.05,
        help="Minimum score to accept a detection",
    )
    iou = st.sidebar.slider(
        "NMS IoU threshold", 0.10, 1.0, 0.45, 0.05,
        help="IoU threshold for non-max suppression",
    )

    # ── Camera ─────────────────────────────────────────
    st.sidebar.markdown("""
    <div class="sb-heading" style="margin-top:18px;">
        <i class="ri-camera-line"></i> Camera
    </div>""", unsafe_allow_html=True)

    cam_idx = st.sidebar.number_input(
        "Device index", min_value=0, max_value=10, value=0,
        help="0 = default webcam",
    )
    res_map = {
        "640 × 480":  (640, 480),
        "800 × 600":  (800, 600),
        "1280 × 720": (1280, 720),
    }
    res_lbl = st.sidebar.selectbox(
        "Resolution", list(res_map.keys()), index=0,
        label_visibility="visible",
    )
    resolution = res_map[res_lbl]

    # ── Display ────────────────────────────────────────
    st.sidebar.markdown("""
    <div class="sb-heading" style="margin-top:18px;">
        <i class="ri-settings-3-line"></i> Display
    </div>""", unsafe_allow_html=True)

    show_fps = st.sidebar.checkbox("FPS overlay", value=True)
    flip     = st.sidebar.checkbox("Mirror camera", value=True)
    max_fps  = st.sidebar.slider("Frame-rate cap", 5, 60, 30)

    # ── System ─────────────────────────────────────────
    st.sidebar.markdown("""
    <div class="sb-heading" style="margin-top:18px;">
        <i class="ri-information-line"></i> System
    </div>""", unsafe_allow_html=True)

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
        "model":       model_map[label],
        "confidence":  conf,
        "iou":         iou,
        "camera_index":int(cam_idx),
        "resolution":  resolution,
        "show_fps":    show_fps,
        "flip_camera": flip,
        "max_fps":     max_fps,
    }


# ============================================================
# PHASE 10: DETECTION LOOP
# ============================================================
def run_detection(config, model):
    # ── Layout ─────────────────────────────────────────
    col_feed, col_panel = st.columns([3, 1], gap="large")

    with col_feed:
        # Action bar above feed
        st.markdown("""
        <div class="action-bar">
            <div class="action-bar-left">
                <div class="action-bar-title">Live Detection Feed</div>
                <div class="action-bar-sub">
                    Real-time inference — adjust settings in the sidebar
                </div>
            </div>
        </div>""", unsafe_allow_html=True)
        feed_ph = st.empty()

    with col_panel:
        st.markdown("""
        <div style="height:52px;"></div>
        """, unsafe_allow_html=True)  # align with feed
        fps_ph     = st.empty()
        alert_ph   = st.empty()
        detlist_ph = st.empty()
        session_ph = st.empty()

    # Stop button — right-aligned below feed/panel
    _, _, btn_col = st.columns([2, 1, 1])
    with btn_col:
        stop = st.button("Stop Detection", type="secondary", key="stop_btn")
        if stop:
            st.session_state["running"] = False

    # ── Camera init ────────────────────────────────────
    cap = cv2.VideoCapture(config["camera_index"], cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(config["camera_index"])
    if not cap.isOpened():
        st.error(f"Camera {config['camera_index']} unavailable.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config["resolution"][0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["resolution"][1])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, config["max_fps"])

    fps_ctr    = FPSCounter(30)
    frame_n    = 0
    total_dets = 0
    t_start    = time.perf_counter()
    min_ft     = 1.0 / config["max_fps"]
    failures   = 0
    MAX_FAIL   = 10

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
                    frame, model, config["confidence"], config["iou"]
                )
            except Exception as exc:
                st.warning(f"Inference error: {exc}")
                ann, dets = frame, []
                det_counts, human = collections.Counter(), False

            fps_ctr.tick()
            frame_n    += 1
            total_dets += len(dets)
            cur_fps     = fps_ctr.fps()
            ann         = to_cpu_mat(ann)

            # Overlays on frame
            if config["show_fps"]:
                cv2.putText(ann,
                    f"FPS  {cur_fps:.1f}",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 212, 126), 2, cv2.LINE_AA)
                cv2.putText(ann,
                    config["model"].upper(),
                    (12, 48), cv2.FONT_HERSHEY_SIMPLEX,
                    0.36, (140, 150, 165), 1, cv2.LINE_AA)

            ts = time.strftime("%H:%M:%S")
            fw = ann.shape[1]
            (tsw, _), _ = cv2.getTextSize(
                ts, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.putText(ann, ts,
                (fw - tsw - 10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (120, 120, 145), 1, cv2.LINE_AA)

            _render_image(feed_ph, frame_to_pil(ann), "")

            # ── Performance panel ───────────────────────
            fps_ph.markdown(f"""
            <div class="sp">
                <div class="sp-head">
                    <div class="sp-icon green"><i class="ri-speed-line"></i></div>
                    <span class="sp-title">Performance</span>
                </div>
                <div class="fps-big">
                    <span class="fps-num">{cur_fps:.1f}</span>
                    <span class="fps-unit">fps</span>
                </div>
                <div class="fps-sub">Frame &nbsp;#{frame_n:,}</div>
            </div>""", unsafe_allow_html=True)

            # ── Alert panel ─────────────────────────────
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

            # ── Detections list ─────────────────────────
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
                        <div class="sp-icon blue">
                            <i class="ri-focus-3-line"></i>
                        </div>
                        <span class="sp-title">Detections</span>
                    </div>
                    {rows_html}
                </div>""", unsafe_allow_html=True)
            else:
                detlist_ph.markdown("""
                <div class="sp">
                    <div class="sp-head">
                        <div class="sp-icon blue">
                            <i class="ri-focus-3-line"></i>
                        </div>
                        <span class="sp-title">Detections</span>
                    </div>
                    <div class="det-empty">
                        <i class="ri-radar-line"></i>
                        <span>Scanning scene…</span>
                    </div>
                </div>""", unsafe_allow_html=True)

            # ── Session panel ───────────────────────────
            elapsed = time.perf_counter() - t_start
            m, s = divmod(int(elapsed), 60)
            avg  = total_dets / max(frame_n, 1)
            session_ph.markdown(f"""
            <div class="sp">
                <div class="sp-head">
                    <div class="sp-icon violet">
                        <i class="ri-bar-chart-2-line"></i>
                    </div>
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

            # Frame-rate cap
            sleep_t = max(0.0, min_ft - (time.perf_counter() - t0))
            if sleep_t > 0:
                time.sleep(sleep_t)

    except Exception as exc:
        st.error(f"Detection error: {exc}")
        raise
    finally:
        cap.release()
        st.session_state["running"] = False
        st.info("Camera released — session ended.")


# ============================================================
# PHASE 11: MAIN
# ============================================================
def main():
    # Initialise state
    if "running" not in st.session_state:
        st.session_state["running"] = False

    # ── Sidebar ────────────────────────────────────────
    config = render_sidebar()

    # ── Model load ─────────────────────────────────────
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

    # ── Top bar ────────────────────────────────────────
    status_dot = "live" if is_live else "idle"
    status_txt = "Live" if is_live else "Idle"

    st.markdown(f"""
    <div class="topbar">
        <div class="topbar-brand">
            <div class="brand-mark">
                <i class="ri-eye-line"></i>
            </div>
            <div class="brand-text">
                <div class="brand-name">Vision<span>AI</span></div>
                <div class="brand-tagline">Real-time object detection — YOLOv8</div>
            </div>
        </div>
        <div class="topbar-status">
            <div class="status-dot {status_dot}"></div>
            {status_txt}
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Metric strip ───────────────────────────────────
    st.markdown(f"""
    <div class="metric-strip">
        <div class="metric-tile">
            <div class="metric-tile-header">
                <span class="metric-tile-label">Model</span>
                <div class="metric-tile-icon blue">
                    <i class="ri-cpu-line"></i>
                </div>
            </div>
            <div class="metric-tile-value">{config['model'].upper()}</div>
            <div class="metric-tile-sub">YOLOv8 variant</div>
        </div>
        <div class="metric-tile">
            <div class="metric-tile-header">
                <span class="metric-tile-label">Compute</span>
                <div class="metric-tile-icon {dev_cls}">
                    <i class="ri-flashlight-line"></i>
                </div>
            </div>
            <div class="metric-tile-value {dev_cls}">{dev_label}</div>
            <div class="metric-tile-sub">Inference device</div>
        </div>
        <div class="metric-tile">
            <div class="metric-tile-header">
                <span class="metric-tile-label">Classes</span>
                <div class="metric-tile-icon violet">
                    <i class="ri-price-tag-3-line"></i>
                </div>
            </div>
            <div class="metric-tile-value">{len(model.names)}</div>
            <div class="metric-tile-sub">Detectable objects</div>
        </div>
        <div class="metric-tile">
            <div class="metric-tile-header">
                <span class="metric-tile-label">Camera</span>
                <div class="metric-tile-icon amber">
                    <i class="ri-camera-line"></i>
                </div>
            </div>
            <div class="metric-tile-value">ID&nbsp;{config['camera_index']}</div>
            <div class="metric-tile-sub">{config['resolution'][0]} × {config['resolution'][1]}</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Classes expander ───────────────────────────────
    with st.expander("Detectable classes", expanded=False):
        tags = ""
        for cls in model.names.values():
            t_cls = "person" if cls == "person" else ""
            tags += f'<span class="class-pill {t_cls}">{cls}</span>'
        st.markdown(
            f'<div style="padding:12px 4px 4px;line-height:2.4;">{tags}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="hdivider"/>', unsafe_allow_html=True)

    # ── Action area ────────────────────────────────────
    if not is_live:
        # Button centered, fixed width
        left_sp, btn_c, right_sp = st.columns([2, 1, 2])
        with btn_c:
            if st.button("Start Detection", type="primary", key="start_btn"):
                st.session_state["running"] = True
                st.rerun()

        # Hero idle state
        st.markdown("""
        <div class="hero-wrap">
            <div class="hero-icon-ring">
                <i class="ri-camera-lens-line"></i>
            </div>
            <div class="hero-heading">Ready to Detect</div>
            <div class="hero-body">
                Configure your model and camera in the sidebar,
                then press <strong style="color:var(--green);">Start Detection</strong>
                to begin real-time inference.
            </div>
            <div class="hero-chips">
                <div class="h-chip green">
                    <i class="ri-speed-line"></i>
                    Real-time inference
                </div>
                <div class="h-chip blue">
                    <i class="ri-focus-3-line"></i>
                    80+ object classes
                </div>
                <div class="h-chip red">
                    <i class="ri-user-search-line"></i>
                    Human detection alert
                </div>
                <div class="h-chip amber">
                    <i class="ri-line-chart-line"></i>
                    Live session analytics
                </div>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        run_detection(config, model)


if __name__ == "__main__":
    main()