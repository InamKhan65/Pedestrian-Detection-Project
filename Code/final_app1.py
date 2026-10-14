# ╔══════════════════════════════════════════════════════════════════╗
# ║                 BYTETRACK — PRODUCTION BUILD                       ║
# ║  Self-contained single-file build. Auto-creates dark theme        ║
# ║  config on first run. No external files needed.                   ║
# ╚══════════════════════════════════════════════════════════════════╝

# ✅ AUTO-CONFIG: Dark theme setup (runs BEFORE any Streamlit imports)
# Creates .streamlit/config.toml next to this script on first run.
# This eliminates the light→dark flicker caused by st.rerun() during
# video playback. On the very first run, a brief flash may appear;
# on every subsequent run (and every st.rerun), the native dark theme
# is loaded before the script executes — zero flash.
import os as _os, sys as _sys
_TOML_CONTENT = '[theme]\nbase = "dark"\nprimaryColor = "#3fb950"\nbackgroundColor = "#0d1117"\nsecondaryBackgroundColor = "#161b22"\ntextColor = "#c9d1d9"\nfont = "sans serif"\n\n[server]\nmaxUploadSize = 500\n'
_config_written = False
# Try __file__ first (works on Windows and normal Python)
try:
    _script_dir = _os.path.dirname(_os.path.abspath(__file__))
    _cfg = _os.path.join(_script_dir, ".streamlit", "config.toml")
    if not _os.path.exists(_cfg):
        _os.makedirs(_os.path.dirname(_cfg), exist_ok=True)
        with open(_cfg, "w", encoding="utf-8") as _fh:
            _fh.write(_TOML_CONTENT)
        _config_written = True
except Exception:
    pass
# Fallback: use CWD (works when Streamlit changes to script directory)
if not _config_written:
    try:
        _cfg2 = _os.path.join(_os.getcwd(), ".streamlit", "config.toml")
        if not _os.path.exists(_cfg2):
            _os.makedirs(_os.path.dirname(_cfg2), exist_ok=True)
            with open(_cfg2, "w", encoding="utf-8") as _fh:
                _fh.write(_TOML_CONTENT)
            _config_written = True
    except Exception:
        pass
del _os, _sys, _TOML_CONTENT, _config_written

import streamlit as st
import cv2
import gc
import numpy as np
import tempfile
import os
import time
import json
import io
import random
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from ultralytics import YOLO
import threading

# ✅ NumPy compatibility — np.float was removed in NumPy 1.24+
# Use np.float64 as the replacement (safe across all versions)

# ✅ Page Configuration (MUST be first Streamlit command)
st.set_page_config(
    page_title="ByteTrack — Pedestrian Detection & Tracking",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ✅ PROGRAMMATIC DARK THEME (runtime — sets dark base immediately so
#    even the first render is dark, no server restart needed. The
#    config.toml above handles persistence across server restarts.)
try:
    from streamlit import config as _st_config
    _st_config.set_option("theme.base", "dark")
    _st_config.set_option("theme.backgroundColor", "#0d1117")
    _st_config.set_option("theme.secondaryBackgroundColor", "#161b22")
    _st_config.set_option("theme.textColor", "#c9d1d9")
except Exception:
    pass

# ╔══════════════════════════════════════════════════════════════════╗
# ║         ANTI-FLASH CSS (injected BEFORE any widgets)           ║
# ║  This prevents the light→dark flash on every st.rerun().       ║
# ║  Injected at line ~78, before ANY Streamlit UI elements.        ║
# ╚══════════════════════════════════════════════════════════════════╝
st.markdown("""<style>
/* ── CRITICAL: Force dark on every element immediately ── */
html, body, [class*="stApp"], .stApp, div[data-testid="stMain"] {
    background-color: #0d1117 !important;
    color: #c9d1d9 !important;
}
/* ── Kill Streamlit's default pink primary color ── */
:root {
    --primary-color: #3fb950 !important;
}
/* ── Kill pink line on st.image containers ── */
[data-testid="stImage"] {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}
[data-testid="stImage"] img,
[data-testid="stImage"] div,
[data-testid="stImage"] figure {
    border: none !important;
    outline: none !important;
}
/* ── Kill pink tab indicator ── */
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-highlight"] + * {
    background-color: #3fb950 !important;
}
</style>""", unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════════════╗
# ║                   DARK THEME CSS                           ║
# ╚══════════════════════════════════════════════════════════════════╝

DARK_THEME_CSS = """
/* ── Global Overrides ── */
.stApp { background-color: var(--color-bg-primary); color: var(--color-text-secondary); font-family: var(--font-family); }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
/* Custom footer overrides Streamlit's hidden footer */
.app-footer { visibility: visible !important; }
/* ── Header: collapse to 0 height ──
   CRITICAL (Streamlit 1.56): The sidebar expand button (stExpandSidebarButton)
   lives INSIDE stToolbar, which is INSIDE header.stAppHeader. We CANNOT use
   visibility:hidden on any ancestor of the expand button, because
   visibility:hidden cannot be overridden by children (CSS spec).
   Instead, use height:0 + overflow:hidden to visually collapse, then let
   position:fixed on the expand button escape the clipping. */
header.stAppHeader { overflow: visible !important; position: relative !important; height: 0 !important; min-height: 0 !important; padding: 0 !important; margin: 0 !important; background: transparent !important; border: none !important; }
/* Toolbar — collapse with height:0, overflow:hidden clips unwanted content.
   NO visibility:hidden here! The expand button must remain visible. */
[data-testid="stToolbar"] {
    height: 0 !important;
    min-height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    overflow: hidden !important;
}
/* Hide individual toolbar children we don't want (actions, deploy, menu) */
[data-testid="stToolbarActions"],
[data-testid="stAppDeployButton"] {
    display: none !important;
}
/* Sidebar EXPAND button (appears when sidebar is collapsed) — position:fixed
   escapes the height:0 + overflow:hidden clipping from ancestor stToolbar. */
[data-testid="stExpandSidebarButton"] {
    visibility: visible !important;
    opacity: 1 !important;
    position: fixed !important;
    top: 10px !important;
    left: 10px !important;
    z-index: 99999 !important;
    width: 36px !important;
    height: 36px !important;
    cursor: pointer !important;
    background: #21262d !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    padding: 0 !important;
    overflow: visible !important;
    pointer-events: auto !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: all 0.2s ease !important;
}
[data-testid="stExpandSidebarButton"]:hover {
    border-color: #58a6ff !important;
    background: #30363d !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5) !important;
}
[data-testid="stExpandSidebarButton"]:hover span {
    color: #58a6ff !important;
}
/* Also handle legacy stSidebarCollapsedControl for older Streamlit versions */
[data-testid="stSidebarCollapsedControl"] {
    visibility: visible !important;
    opacity: 1 !important;
    position: fixed !important;
    top: 10px !important;
    left: 10px !important;
    z-index: 99999 !important;
    width: 36px !important;
    height: 36px !important;
    cursor: pointer !important;
    background: #21262d !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    padding: 0 !important;
    overflow: visible !important;
    pointer-events: auto !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: all 0.2s ease !important;
}
[data-testid="stSidebarCollapsedControl"]:hover {
    border-color: #58a6ff !important;
    background: #30363d !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5) !important;
}
[data-testid="stSidebarCollapsedControl"] > div,
[data-testid="stSidebarCollapsedControl"] > button,
[data-testid="stSidebarCollapsedControl"] > span {
    visibility: hidden !important;
    font-size: 0 !important;
}
[data-testid="stSidebarCollapsedControl"]::before {
    content: '' !important;
    position: absolute !important;
    width: 16px !important;
    height: 2px !important;
    background-color: #c9d1d9 !important;
    border-radius: 1px !important;
    box-shadow: 0 5px 0 #c9d1d9, 0 10px 0 #c9d1d9 !important;
    transition: all 0.2s ease !important;
}
[data-testid="stSidebarCollapsedControl"]:hover::before {
    background-color: #58a6ff !important;
    box-shadow: 0 5px 0 #58a6ff, 0 10px 0 #58a6ff !important;
}
.block-container { padding-top: 0.5rem; padding-bottom: 1rem; max-width: 1400px; }

:root {
    /* ── Color Palette ── */
    --color-bg-primary: #0d1117;
    --color-bg-secondary: #161b22;
    --color-bg-tertiary: #21262d;
    --color-border-default: #30363d;
    --color-border-muted: #21262d;
    --color-border-subtle: #484f58;
    --color-text-primary: #f0f6fc;
    --color-text-secondary: #c9d1d9;
    --color-text-muted: #8b949e;
    --color-text-faint: #484f58;
    --color-accent-blue: #58a6ff;
    --color-accent-blue-deep: #1f6feb;
    --color-accent-green: #3fb950;
    --color-accent-green-dark: #238636;
    --color-accent-green-mid: #2ea043;
    --color-accent-red: #f85149;
    --color-accent-red-dark: #da3633;
    --color-accent-purple: #a371f7;
    --color-accent-purple-dark: #8957e5;
    --color-accent-yellow: #e3b341;
    --color-accent-yellow-dark: #d29922;
    /* ── Type Scale System ── */
    --text-display: 32px;
    --text-title: 18px;
    --text-subtitle: 15px;
    --text-body: 14px;
    --text-caption: 12px;
    --text-micro: 10px;
    --font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji';
    --weight-bold: 800;
    --weight-semibold: 700;
    --weight-medium: 600;
    --weight-normal: 500;
    --weight-regular: 400;
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --transition-fast: 0.15s ease;
    --transition-base: 0.2s ease;
    --transition-smooth: 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
section[data-testid="stSidebar"] .stMarkdown { color: #c9d1d9; }
.sidebar-section { background-color: #0d1117; border: 1px solid #30363d; border-radius: 12px; padding: 16px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.08); }

/* ── Icon System ── */
.icon { display: inline-flex; align-items: center; justify-content: center; width: 20px; height: 20px; font-size: 14px; font-weight: 700; border-radius: 4px; flex-shrink: 0; }
.icon-enter { color: #3fb950; }
.icon-exit { color: #f85149; }
.icon-net { color: #58a6ff; }
.icon-unique { color: #a371f7; }
.icon-current { color: #e3b341; }
.icon-conf { color: #58a6ff; }
.icon-line { color: #8b949e; }
.icon-speed { color: #e3b341; }
.icon-peak { color: #a371f7; }
.icon-event { color: #c9d1d9; }
.icon-model { color: #58a6ff; }
.icon-track { color: #a371f7; }
.icon-info { color: #8b949e; }
.icon-system { color: #c9d1d9; }
.icon-input { color: #58a6ff; }
.icon-play { color: #3fb950; }
.icon-export { color: #e3b341; }

/* Section header icon dots */
.section-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 8px; flex-shrink: 0; }
.dot-blue { background-color: #58a6ff; }
.dot-green { background-color: #3fb950; }
.dot-purple { background-color: #a371f7; }
.dot-yellow { background-color: #e3b341; }
.dot-red { background-color: #f85149; }
.dot-gray { background-color: #8b949e; }

/* Sidebar section h3 update — remove emoji, use dot prefix */
.sidebar-section h3 { 
    color: #58a6ff; 
    font-size: var(--text-caption); 
    font-weight: var(--weight-semibold); 
    text-transform: uppercase; 
    letter-spacing: 0.8px; 
    margin-bottom: 12px; 
    padding-bottom: 8px; 
    border-bottom: 1px solid #21262d;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Header ── */
.app-header { background: linear-gradient(135deg, #161b22 0%, #1a2332 50%, #161b22 100%); border: 1px solid #30363d; border-radius: 16px; padding: 20px 28px; margin-bottom: 20px; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.08); }
.app-header .logo-area { display: flex; align-items: center; gap: 16px; }
.app-header .logo-icon { width: 48px; height: 48px; background: linear-gradient(135deg, #58a6ff, #1f6feb); border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 24px; box-shadow: 0 4px 12px rgba(88, 166, 255, 0.3); }
.app-header h1 { color: #f0f6fc; font-size: 26px; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
.app-header .subtitle { color: #8b949e; font-size: 13px; margin-top: 2px; font-weight: 400; }
.app-header .status-badge { background-color: #238636; color: white; padding: 6px 16px; border-radius: 20px; font-size: 12px; font-weight: 600; letter-spacing: 0.5px; }
.app-header .status-badge.paused { background-color: #d29922; }
.app-header .status-badge.running { background-color: #238636; animation: pulse 2s infinite; }

/* ── Metric Cards ── */
.metric-cards-row { display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; margin-bottom: 20px; }
.metric-card { background-color: #161b22; border: 1px solid #30363d; border-radius: 14px; padding: 18px 20px; text-align: center; transition: all 0.3s ease; position: relative; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.08); }
.metric-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; border-radius: 14px 14px 0 0; }
.metric-card.card-in::before { background: linear-gradient(90deg, #238636, #2ea043); }
.metric-card.card-out::before { background: linear-gradient(90deg, #da3633, #f85149); }
.metric-card.card-net::before { background: linear-gradient(90deg, #1f6feb, #58a6ff); }
.metric-card.card-unique::before { background: linear-gradient(90deg, #8957e5, #a371f7); }
.metric-card.card-current::before { background: linear-gradient(90deg, #d29922, #e3b341); }
.metric-card .metric-icon { font-size: 22px; margin-bottom: 6px; opacity: 0.9; }
.metric-card .metric-value { font-size: 32px; font-weight: 800; line-height: 1; margin-bottom: 4px; font-variant-numeric: tabular-nums; }
.metric-card .metric-label { font-size: 12px; color: #8b949e; font-weight: 500; text-transform: uppercase; letter-spacing: 0.8px; }
.metric-card.card-in .metric-value { color: #3fb950; }
.metric-card.card-out .metric-value { color: #f85149; }
.metric-card.card-net .metric-value { color: #58a6ff; }
.metric-card.card-unique .metric-value { color: #a371f7; }
.metric-card.card-current .metric-value { color: #e3b341; }

/* ── Minimal Metric Cards for Welcome ── */
.metric-cards-minimal { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-bottom: 20px; max-width: 640px; margin-left: auto; margin-right: auto; }

/* ── Video Display with Gradient Border ── */
.video-display-area { 
    background-color: #161b22; 
    border: 2px solid transparent;
    border-radius: 14px; 
    padding: 16px; 
    position: relative; 
    overflow: hidden;
    background-clip: padding-box;
    background-origin: border-box;
    background-image: 
        linear-gradient(#161b22, #161b22),
        linear-gradient(135deg, #58a6ff, #a371f7, #3fb950, #e3b341, #58a6ff);
    background-size: 100% 100%, 300% 300%;
    animation: gradientBorder 8s ease infinite;
}
@keyframes gradientBorder {
    0% { background-position: 0% 0%, 0% 50%; }
    33% { background-position: 0% 0%, 100% 50%; }
    66% { background-position: 0% 0%, 50% 100%; }
    100% { background-position: 0% 0%, 0% 50%; }
}
.video-display-area .area-label { color: #8b949e; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }
.video-display-area .area-label .dot { width: 8px; height: 8px; border-radius: 50%; background-color: #238636; animation: pulse 2s infinite; }
.video-display-area .area-label .dot.paused { background-color: #d29922; animation: none; }

/* ── Elevation System ── */
.elevation-0 { box-shadow: none; }
.elevation-1 { box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.08); }
.elevation-2 { box-shadow: 0 4px 12px rgba(0,0,0,0.25), 0 2px 4px rgba(0,0,0,0.12); }
.elevation-3 { box-shadow: 0 8px 24px rgba(0,0,0,0.4), 0 4px 8px rgba(0,0,0,0.2); }

/* ── Stats Panel ── */
.stats-panel { background-color: #161b22; border: 1px solid #30363d; border-radius: 14px; padding: 20px; height: 100%; box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.08); }
.stats-panel h3 { color: #f0f6fc; font-size: 15px; font-weight: 600; margin-bottom: 16px; padding-bottom: 10px; border-bottom: 1px solid #21262d; display: flex; align-items: center; gap: 8px; }
.stats-item { display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #21262d; }
.stats-item:last-child { border-bottom: none; }
.stats-item .stats-label { color: #8b949e; font-size: 13px; display: flex; align-items: center; gap: 8px; }
.stats-item .stats-value { font-size: 16px; font-weight: 700; font-variant-numeric: tabular-nums; }

/* ── Welcome State ── */
.welcome-state { background-color: #161b22; border: 1px solid #30363d; border-radius: 14px; padding: 60px 40px; text-align: center; }
.welcome-state .welcome-icon { font-size: 56px; margin-bottom: 16px; }
.welcome-state h2 { color: #f0f6fc; font-size: 22px; font-weight: 600; margin-bottom: 10px; }
.welcome-state p { color: #8b949e; font-size: 15px; max-width: 500px; margin: 0 auto; line-height: 1.6; }

/* ── Sidebar widgets (unified rules in responsive section below) ── */

/* ── Hide Streamlit widget labels EXCEPT checkboxes ── */
/* Target specific widgets only — checkboxes use stCheckbox which has no separate label row */
.stSelectbox [data-testid="stWidgetLabelRow"] { display: none !important; }
.stSlider [data-testid="stWidgetLabelRow"] { display: none !important; }
.stFileUploader [data-testid="stWidgetLabelRow"] { display: none !important; }

/* ── Slider & Toggle ── */
.slider-label { color: #8b949e; font-size: 13px; font-weight: 500; display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
.slider-label .slider-val { color: #58a6ff; font-weight: 700; font-size: 13px; }
.toggle-row { display: flex; justify-content: space-between; align-items: center; padding: 6px 0; }

/* ── Summary Panel ── */
.summary-panel { background-color: #161b22; border: 1px solid #30363d; border-radius: 14px; padding: 24px; margin-top: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.08); }
.summary-panel h3 { color: #f0f6fc; font-size: 18px; font-weight: 700; margin-bottom: 20px; display: flex; align-items: center; gap: 10px; }
.summary-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
.summary-card { background-color: #0d1117; border: 1px solid #21262d; border-radius: 12px; padding: 16px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.08); }
.summary-card .sum-icon { font-size: 28px; margin-bottom: 8px; }
.summary-card .sum-value { font-size: 28px; font-weight: 800; margin-bottom: 4px; font-variant-numeric: tabular-nums; }
.summary-card .sum-label { font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; font-weight: 500; }

/* ── Chart Area ── */
.chart-area { background-color: #161b22; border: 1px solid #30363d; border-radius: 14px; padding: 16px; margin-top: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.08); }
.chart-area .area-label { color: #8b949e; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px; display: flex; align-items: center; gap: 8px; }

/* ── Export Buttons Styling ── */
.export-buttons { display: flex; gap: 8px; flex-wrap: wrap; }
.export-btn { display: inline-flex; align-items: center; gap: 6px; padding: 8px 16px; border-radius: 8px; font-size: 13px; font-weight: 600; cursor: pointer; border: 1px solid #30363d; background-color: #21262d; color: #c9d1d9; transition: all 0.2s ease; }
.export-btn:hover { background-color: #30363d; border-color: #58a6ff; }
.export-btn.csv-btn:hover { border-color: #3fb950; color: #3fb950; }
.export-btn.json-btn:hover { border-color: #58a6ff; color: #58a6ff; }
.export-btn.snap-btn:hover { border-color: #e3b341; color: #e3b341; }

/* ── Activity Log Table ── */
.activity-log-table { width: 100%; border-collapse: collapse; margin-top: 8px; }
.activity-log-table th { color: #8b949e; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; padding: 8px 12px; text-align: left; border-bottom: 2px solid #30363d; font-weight: 600; }
.activity-log-table td { color: #c9d1d9; font-size: 12px; padding: 6px 12px; border-bottom: 1px solid #21262d; }
.activity-log-table tr:hover td { background-color: #161b22; }
.event-entered { color: #3fb950; font-weight: 600; }
.event-exited { color: #f85149; font-weight: 600; }

/* ── Tab Navigation Styling ── */
.tab-nav-custom { margin-bottom: 16px; }
.settings-section { 
    background-color: #161b22; 
    border: 1px solid #30363d; 
    border-radius: 14px; 
    padding: 20px; 
    margin-bottom: 16px; 
    box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.08);
}
.settings-section h3 { 
    color: #f0f6fc; 
    font-size: 15px; 
    font-weight: 600; 
    margin-bottom: 14px; 
    padding-bottom: 10px; 
    border-bottom: 1px solid #21262d; 
    display: flex; 
    align-items: center; 
    gap: 8px; 
}
.settings-row { 
    display: flex; 
    justify-content: space-between; 
    align-items: center; 
    padding: 10px 0; 
    border-bottom: 1px solid #21262d; 
}
.settings-row:last-child { border-bottom: none; }
.settings-row .settings-label { color: #8b949e; font-size: 13px; }
.settings-row .settings-label small { display: block; color: #484f58; font-size: 11px; margin-top: 2px; }
.settings-row .settings-value { color: #c9d1d9; font-size: 14px; font-weight: 600; font-variant-numeric: tabular-nums; }
.settings-row .settings-value.on { color: #3fb950; }
.settings-row .settings-value.off { color: #484f58; }
.settings-row .settings-value.paused { color: #d29922; }
.analytics-stat-card { 
    background-color: #161b22; 
    border: 1px solid #30363d; 
    border-radius: 14px; 
    padding: 20px; 
    text-align: center; 
    box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.08);
}
.analytics-stat-card .stat-icon { font-size: 28px; margin-bottom: 8px; }
.analytics-stat-card .stat-value { font-size: 36px; font-weight: 800; margin-bottom: 4px; font-variant-numeric: tabular-nums; }
.analytics-stat-card .stat-label { font-size: 12px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 500; }
.analytics-empty { 
    text-align: center; 
    padding: 80px 40px; 
    color: #8b949e; 
}
.analytics-empty .empty-icon { font-size: 64px; margin-bottom: 16px; opacity: 0.5; }
.analytics-empty h3 { color: #c9d1d9; font-size: 20px; margin-bottom: 8px; }
.analytics-empty p { font-size: 14px; max-width: 400px; margin: 0 auto; line-height: 1.6; }

/* ── Loading Spinner ── */
.loading-overlay { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 60px 20px; text-align: center; }
.loading-spinner { display: flex; gap: 8px; margin-bottom: 20px; }
.loading-spinner .dot { width: 14px; height: 14px; border-radius: 50%; background-color: #58a6ff; animation: loadingBounce 1.4s ease-in-out infinite both; }
.loading-spinner .dot:nth-child(1) { animation-delay: -0.32s; }
.loading-spinner .dot:nth-child(2) { animation-delay: -0.16s; }
.loading-spinner .dot:nth-child(3) { animation-delay: 0s; }
.loading-text { color: #c9d1d9; font-size: 15px; font-weight: 500; }
.loading-subtext { color: #484f58; font-size: 12px; margin-top: 6px; }
@keyframes loadingBounce { 0%, 80%, 100% { transform: scale(0.4); opacity: 0.4; } 40% { transform: scale(1); opacity: 1; } }

/* ── Progress Bar ── */
.progress-container { background-color: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 12px 16px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.08); }
.progress-info { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.progress-info .progress-label { color: #8b949e; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
.progress-info .progress-pct { color: #3fb950; font-size: 14px; font-weight: 700; font-variant-numeric: tabular-nums; }
.progress-bar-track { width: 100%; height: 6px; background-color: #21262d; border-radius: 3px; overflow: hidden; }
.progress-bar-fill { height: 100%; background: linear-gradient(90deg, #238636, #3fb950); border-radius: 3px; transition: width 0.3s ease; }
.progress-meta { display: flex; justify-content: space-between; margin-top: 6px; color: #484f58; font-size: 11px; }

/* ── Streamlit progress override ── */
.stProgress > div > div > div { background-color: #3fb950; }

/* ── Animated Metric Card Pulse ── */
.metric-value-pulse { animation: metricPulse 0.5s ease-out; }
@keyframes metricPulse { 0% { transform: scale(1.15); filter: brightness(1.4); } 100% { transform: scale(1); filter: brightness(1); } }

/* ── Toast Notifications ── */
.toast-container { position: fixed; bottom: 20px; right: 20px; z-index: 9999; display: flex; flex-direction: column-reverse; gap: 8px; max-width: 360px; }
.toast-item { background-color: #161b22; border: 1px solid #30363d; border-left: 4px solid #30363d; border-radius: 10px; padding: 12px 16px; animation: toastSlideIn 0.3s ease-out, toastFadeOut 0.3s ease-in 2.7s forwards; }
.toast-item.success { border-left-color: #3fb950; }
.toast-item.info { border-left-color: #58a6ff; }
.toast-item.warning { border-left-color: #d29922; }
.toast-item.danger { border-left-color: #f85149; }
.toast-item.event { border-left-color: #a371f7; }
.toast-item .toast-title { color: #f0f6fc; font-size: 13px; font-weight: 600; margin-bottom: 2px; }
.toast-item .toast-message { color: #8b949e; font-size: 12px; }
.toast-item .toast-time { color: #484f58; font-size: 10px; margin-top: 4px; }
@keyframes toastSlideIn { from { transform: translateX(100%); opacity: 0; } to { transform: translateX(0); opacity: 1; } }
@keyframes toastFadeOut { from { opacity: 1; } to { opacity: 0; } }

/* ── Hover Effects & Micro-Transitions ── */
.metric-card:hover { transform: translateY(-2px); border-color: #484f58; box-shadow: 0 4px 16px rgba(0,0,0,0.35); }
.summary-card { transition: all 0.2s ease; }
.summary-card:hover { border-color: #484f58; transform: translateY(-1px); box-shadow: 0 4px 16px rgba(0,0,0,0.35); }
.analytics-stat-card { transition: all 0.2s ease; }
.activity-log-table tbody tr { transition: background-color 0.15s ease; }
.activity-log-table tbody tr:hover td { background-color: rgba(33, 38, 45, 0.8) !important; }
.settings-row { transition: background-color 0.15s ease; border-radius: 6px; padding: 10px 8px; }
.settings-row:hover { background-color: #1c2128; }
.sidebar-section { transition: border-color 0.2s ease; }
.sidebar-section:hover { border-color: #484f58; }
.chart-area { transition: border-color 0.2s ease; }
.chart-area:hover { border-color: #484f58; }

/* ── Custom Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d1117; border-radius: 3px; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #484f58; }

/* ── Focus States for Accessibility ── */
.metric-card:focus-within { outline: 2px solid #58a6ff; outline-offset: 2px; }
.sidebar-section:focus-within { outline: 2px solid #58a6ff; outline-offset: 2px; }

/* ── Progress Bar Glow ── */
.progress-bar-fill { box-shadow: 0 0 8px rgba(63, 185, 80, 0.4); }

/* ── Footer ── */
.app-footer { text-align: center; padding: 20px 0 10px 0; color: #484f58; font-size: 12px; border-top: 1px solid #21262d; margin-top: 20px; }
.app-footer a { color: #58a6ff; text-decoration: none; }
.app-footer a:hover { text-decoration: underline; }

/* ── Dark Tab Styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background-color: var(--color-bg-secondary);
    border: 1px solid var(--color-border-default);
    border-radius: var(--radius-md);
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: var(--radius-sm) !important;
    color: var(--color-text-muted) !important;
    font-weight: var(--weight-medium);
    font-size: var(--text-body);
    padding: 8px 20px;
    transition: all var(--transition-base);
    background-color: transparent !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--color-text-secondary) !important;
    background-color: var(--color-bg-tertiary) !important;
}
.stTabs [aria-selected="true"] {
    color: var(--color-accent-blue) !important;
    background-color: var(--color-bg-tertiary) !important;
    font-weight: var(--weight-semibold) !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    background-color: transparent !important;
    height: 0 !important;
}
.stTabs [aria-selected="true"] + [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-highlight"].tab-highlight-active {
    background-color: var(--color-accent-blue) !important;
    height: 2px !important;
    border-radius: 2px;
}
.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}

/* ── Expander Smooth Animation ── */
[data-testid="stExpander"] details {
    transition: all var(--transition-smooth);
    border: 1px solid var(--color-border-default);
    border-radius: var(--radius-md);
    background-color: var(--color-bg-secondary);
    padding: 0;
    overflow: hidden;
}
[data-testid="stExpander"] details[open] {
    border-color: var(--color-border-subtle);
}
[data-testid="stExpander"] summary {
    padding: 12px 16px;
    transition: all var(--transition-base);
    border-radius: var(--radius-md);
}
[data-testid="stExpander"] summary:hover {
    background-color: var(--color-bg-tertiary);
}
[data-testid="stExpander"] .stMarkdown {
    padding: 0 16px 16px 16px;
}

/* ── Slider Thumb Customization ── */
.stSlider [data-testid="stSliderHandle"] {
    background-color: var(--color-accent-blue) !important;
    border: 2px solid var(--color-accent-blue-deep) !important;
    box-shadow: 0 0 8px rgba(88, 166, 255, 0.35) !important;
    transition: all var(--transition-fast) !important;
}
.stSlider [data-testid="stSliderHandle"]:hover {
    box-shadow: 0 0 14px rgba(88, 166, 255, 0.55) !important;
    transform: scale(1.15) !important;
}
.stSlider [data-testid="stSliderHandle"]:active {
    box-shadow: 0 0 18px rgba(88, 166, 255, 0.7) !important;
    transform: scale(1.05) !important;
}

/* ── Unified Button Styling ── */
.stButton > button {
    font-weight: var(--weight-semibold) !important;
    font-size: var(--text-caption) !important;
    letter-spacing: 0.3px;
    border-radius: var(--radius-sm) !important;
    transition: all var(--transition-base) !important;
    border: 1px solid var(--color-border-default) !important;
    text-transform: none;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 3px 10px rgba(0,0,0,0.3);
    border-color: var(--color-accent-blue) !important;
}
.stButton > button:active {
    transform: translateY(0px);
    box-shadow: none;
}
.stButton > button:disabled,
.stButton > button[disabled] {
    opacity: 0.4;
    cursor: not-allowed;
    transform: none !important;
    box-shadow: none !important;
    border-color: var(--color-border-muted) !important;
}

/* ── Download Button Styling  ── */
.stDownloadButton > button {
    font-weight: var(--weight-semibold) !important;
    font-size: var(--text-caption) !important;
    border-radius: var(--radius-sm) !important;
    transition: all var(--transition-base) !important;
    border: 1px solid var(--color-border-default) !important;
    background-color: var(--color-bg-tertiary) !important;
    color: var(--color-text-secondary) !important;
}
.stDownloadButton > button:hover {
    border-color: var(--color-accent-green) !important;
    color: var(--color-accent-green) !important;
    transform: translateY(-1px);
    box-shadow: 0 3px 10px rgba(63, 185, 80, 0.15);
}
.stDownloadButton > button:active {
    transform: translateY(0);
}
.stDownloadButton > button:disabled,
.stDownloadButton > button[disabled] {
    opacity: 0.35 !important;
    cursor: not-allowed !important;
    transform: none !important;
    box-shadow: none !important;
    border-color: var(--color-border-muted) !important;
    background-color: var(--color-bg-primary) !important;
    color: var(--color-text-faint) !important;
}

/* ── Checkbox Styling ── */
.stCheckbox > label {
    color: var(--color-text-secondary);
    font-size: var(--text-body);
    transition: color var(--transition-fast);
    font-weight: var(--weight-normal);
}
.stCheckbox > label:hover {
    color: var(--color-accent-blue);
}

/* ── Accessibility — prefers-reduced-motion ── */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation: none !important;
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition: none !important;
        transition-duration: 0.01ms !important;
        scroll-behavior: auto !important;
    }
    /* Restore non-entrance animations for essential feedback */
    .toast-item { animation: none !important; }
}

/* ── Accessibility — focus-visible ── */
*:focus-visible {
    outline: 2px solid #58a6ff !important;
    outline-offset: 2px !important;
    border-radius: 4px;
}
button:focus-visible {
    outline: 2px solid #58a6ff !important;
    outline-offset: 2px !important;
}

/* ── Accessibility — prefers-contrast: high ── */
@media (prefers-contrast: high) {
    .metric-card, .stats-panel, .summary-panel, .settings-section, .chart-area, .sidebar-section {
        border-color: #8b949e !important;
    }
    .metric-label, .stats-label, .settings-label, .activity-log-table th {
        color: #c9d1d9 !important;
    }
}

/* ── Screen Reader Utility ── */
.sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border-width: 0;
}

/* ── Controls Bar Responsive ── */
@media (max-width: 900px) {
    .progress-meta { flex-wrap: wrap; gap: 4px; }
    .controls-bar-inner { flex-wrap: wrap; gap: 6px !important; }
    .controls-bar-stats { flex-wrap: wrap; gap: 8px !important; }
}
@media (max-width: 600px) {
    .controls-bar-stats span { font-size: 11px !important; }
}

/* ── Sidebar Widgets (unified) ── */
.stSelectbox > div > div {
    background-color: var(--color-bg-primary) !important;
    color: var(--color-text-secondary);
    border-color: var(--color-border-default);
    border-radius: var(--radius-sm);
    transition: border-color var(--transition-base);
    font-family: var(--font-family);
}
.stSelectbox > div > div:hover,
.stSelectbox > div > div:focus-within {
    border-color: var(--color-accent-blue) !important;
}
.stFileUploader > div > div {
    background-color: var(--color-bg-primary);
    border-color: var(--color-border-default);
    border-radius: var(--radius-sm);
    transition: border-color var(--transition-base);
}
.stFileUploader > div > div:hover {
    border-color: var(--color-accent-blue);
}

/* ── Styled Sidebar Dividers ── */
.stSidebar hr, .stSidebar [data-testid="stHorizontalBlock"] + hr {
    border: none !important;
    border-top: 1px solid var(--color-border-muted) !important;
    margin: 4px 0 !important;
}

/* ── Welcome Quick-Action Buttons ── */
/* Style the welcome state buttons to match the dark theme consistently */

/* ── Analytics Stat Cards Responsive ── */
@media (max-width: 768px) {
    .analytics-grid-4 { grid-template-columns: repeat(2, 1fr) !important; }
}
@media (max-width: 480px) {
    .analytics-grid-4 { grid-template-columns: 1fr !important; }
}

/* ── Sidebar Toggle Tooltip ── */
[data-testid="stSidebarCollapsedControl"]::after {
    content: 'Toggle sidebar';
    position: absolute; left: 44px; top: 50%; transform: translateY(-50%);
    background: var(--color-bg-tertiary); color: var(--color-text-secondary);
    padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 500;
    white-space: nowrap; pointer-events: none; opacity: 0; transition: opacity var(--transition-base);
    border: 1px solid var(--color-border-default); box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    z-index: 100000;
}
[data-testid="stSidebarCollapsedControl"]:hover::after {
    opacity: 1;
}

/* ── Video/Image Stats Column Fix ── */
.stats-col-spacer { display: none; }
@media (min-width: 769px) {
    .stats-col-spacer { display: block; }
}

/* ── Toast Mobile Fix ── */
@media (max-width: 600px) {
    .toast-container { bottom: 10px !important; right: 10px !important; left: 10px !important; max-width: calc(100% - 20px) !important; }
    .toast-item { padding: 10px 12px !important; }
    .toast-item .toast-title { font-size: 12px !important; }
    .toast-item .toast-message { font-size: 11px !important; }
}

/* ── Entrance Animation (fade-in) ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}
.app-header { animation: fadeInUp 0.4s ease-out; }
.metric-cards-row, .metric-cards-minimal { animation: fadeInUp 0.4s ease-out 0.1s both; }
.video-display-area { animation: fadeInUp 0.4s ease-out 0.15s both; }
.stats-panel { animation: fadeInUp 0.4s ease-out 0.2s both; }
.welcome-state { animation: fadeInUp 0.5s ease-out 0.1s both; }
.controls-bar-inner { animation: fadeInUp 0.3s ease-out 0.05s both; }
.progress-container { animation: fadeInUp 0.3s ease-out 0.05s both; }
.summary-panel { animation: fadeInUp 0.4s ease-out 0.15s both; }
.chart-area { animation: fadeInUp 0.4s ease-out 0.2s both; }
.settings-section { animation: fadeInUp 0.4s ease-out; }
.analytics-stat-card { animation: fadeInUp 0.4s ease-out; }
.analytics-empty { animation: fadeInUp 0.5s ease-out; }
.sidebar-section { animation: fadeInUp 0.3s ease-out; }
/* .toast-item uses its own toastSlideIn/toastFadeOut — NOT fadeInUp */

/* ── File Uploader Custom Styling ── */
.stFileUploader > div > div {
    border: 2px dashed var(--color-border-subtle) !important;
    transition: all var(--transition-base) !important;
}
.stFileUploader > div > div:hover {
    border-color: var(--color-accent-blue) !important;
    background-color: rgba(88, 166, 255, 0.04) !important;
}
.stFileUploader > div[data-testid="stFileUploaderDropzone"] {
    background-color: var(--color-bg-primary) !important;
}
.stFileUploader [data-testid="stFileUploaderDropzoneInput"] {
    color: var(--color-text-muted) !important;
}
.stFileUploader [data-testid="stFileUploaderDropzoneInput"]::after {
    content: ' or drag and drop';
    color: var(--color-text-faint);
}

/* ── Activity Log Zebra Striping ── */
.activity-log-table tbody tr:nth-child(even) td {
    background-color: rgba(22, 27, 34, 0.4);
}
.activity-log-table tbody tr:nth-child(odd) td {
    background-color: transparent;
}

/* ── Theme st.success() and st.error() Banners ── */
.stSuccess {
    background-color: rgba(63, 185, 80, 0.12) !important;
    border: 1px solid rgba(63, 185, 80, 0.3) !important;
    border-radius: var(--radius-sm) !important;
    color: #3fb950 !important;
}
.stError, .stException {
    background-color: rgba(248, 81, 73, 0.12) !important;
    border: 1px solid rgba(248, 81, 73, 0.3) !important;
    border-radius: var(--radius-sm) !important;
    color: #f85149 !important;
}
.stWarning {
    background-color: rgba(227, 179, 65, 0.12) !important;
    border: 1px solid rgba(227, 179, 65, 0.3) !important;
    border-radius: var(--radius-sm) !important;
    color: #e3b341 !important;
}
.stInfo {
    background-color: rgba(88, 166, 255, 0.12) !important;
    border: 1px solid rgba(88, 166, 255, 0.3) !important;
    border-radius: var(--radius-sm) !important;
    color: #58a6ff !important;
}

/* ── Animation ── */
@keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.4; } 100% { opacity: 1; } }

/* ── Enhanced Responsive ── */
/* Tablet */
@media (max-width: 1024px) {
    .metric-cards-row { grid-template-columns: repeat(3, 1fr); }
    .summary-grid { grid-template-columns: repeat(2, 1fr); }
    .export-buttons { flex-wrap: wrap; }
    .video-display-area { padding: 12px; }
}

/* Small tablet */
@media (max-width: 768px) {
    .app-header { padding: 14px 20px; }
    .app-header h1 { font-size: 20px; }
    .app-header .logo-icon { width: 40px; height: 40px; font-size: 20px; }
    .app-header .status-badge { font-size: 11px; padding: 4px 12px; }
    .settings-row .settings-label { font-size: 12px; }
    .settings-row .settings-value { font-size: 13px; }
    .progress-meta { flex-wrap: wrap; gap: 4px; }
    .video-display-area { padding: 12px; }
}

/* Mobile */
@media (max-width: 600px) {
    .metric-cards-row { grid-template-columns: repeat(2, 1fr); }
    .metric-cards-minimal { grid-template-columns: repeat(2, 1fr); }
    .summary-grid { grid-template-columns: 1fr; }
    .app-header { padding: 12px 16px; }
    .app-header h1 { font-size: 18px; }
    .app-header .subtitle { display: none; }
    .app-header .logo-icon { width: 36px; height: 36px; font-size: 18px; border-radius: 10px; }
    .welcome-state { padding: 40px 20px; }
    .toast-container { max-width: calc(100% - 40px); right: 20px; left: 20px; }
    .progress-meta { font-size: 10px; }
}

/* Very small mobile */
@media (max-width: 480px) {
    .metric-cards-row { gap: 8px; }
    .metric-card { padding: 12px 14px; }
    .metric-card .metric-value { font-size: 24px; }
    .app-header .status-badge { display: none; }
    .app-footer { font-size: 10px; padding: 16px 0 8px 0; }
}
"""

st.markdown(f"<style>{DARK_THEME_CSS}</style>", unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════════════╗
# ║       LIVE VIDEO — MAIN-THREAD AUTO-ADVANCE                     ║
# ║  process_frame() uses st.session_state internally, so it MUST   ║
# ║  run on the main Streamlit thread (not a background thread).     ║
# ║  Approach: store cv2.VideoCapture in session_state, process     ║
# ║  frames on each rerun, use st.rerun() for auto-advance.        ║
# ╚══════════════════════════════════════════════════════════════════╝

# ╔══════════════════════════════════════════════════════╗
# ║                    CUSTOM HEADER                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

# Header badge respects paused state
_running = st.session_state.get("is_running", False)
_paused = st.session_state.get("is_paused", False)
if _running and not _paused:
    badge_class = "running"
    badge_text = "● TRACKING"
elif _running and _paused:
    badge_class = "paused"
    badge_text = "● PAUSED"
else:
    badge_class = ""
    badge_text = "● SYSTEM READY"

st.markdown(f"""
<div class="app-header">
    <div class="logo-area">
        <div class="logo-icon"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><line x1="16.5" y1="16.5" x2="21" y2="21"/></svg></div>
        <div>
            <h1>ByteTrack</h1>
            <div class="subtitle">Real-Time Pedestrian Detection & Multi-Object Tracking</div>
        </div>
    </div>
    <div class="status-badge {badge_class}">{badge_text}</div>
</div>
""", unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════════════╗
# ║               SESSION STATE INITIALIZATION                       ║
# ╚══════════════════════════════════════════════════════════════════╝

if "is_running" not in st.session_state: st.session_state.is_running = False
if "is_paused" not in st.session_state: st.session_state.is_paused = False
if "conf_threshold" not in st.session_state: st.session_state.conf_threshold = 0.4
if "line_y_ratio" not in st.session_state: st.session_state.line_y_ratio = 0.5
if "show_boxes" not in st.session_state: st.session_state.show_boxes = True
if "show_ids" not in st.session_state: st.session_state.show_ids = True
if "show_counting_line" not in st.session_state: st.session_state.show_counting_line = True
# show_stats_overlay removed — redundant with Statistics tab
if "playback_speed" not in st.session_state: st.session_state.playback_speed = 1.0

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.model = None

# ── Session state: tracking variables ──
if "track_history" not in st.session_state: st.session_state.track_history = {}
if "unique_ids" not in st.session_state: st.session_state.unique_ids = set()
if "counted_ids" not in st.session_state: st.session_state.counted_ids = set()
if "in_count" not in st.session_state: st.session_state.in_count = 0
if "out_count" not in st.session_state: st.session_state.out_count = 0

# ── Session state: chart data
if "chart_timestamps" not in st.session_state: st.session_state.chart_timestamps = []
if "chart_in_counts" not in st.session_state: st.session_state.chart_in_counts = []
if "chart_out_counts" not in st.session_state: st.session_state.chart_out_counts = []
if "chart_net_counts" not in st.session_state: st.session_state.chart_net_counts = []
if "chart_current_counts" not in st.session_state: st.session_state.chart_current_counts = []
if "peak_count" not in st.session_state: st.session_state.peak_count = 0
if "session_start_time" not in st.session_state: st.session_state.session_start_time = None
if "line_flash_counter" not in st.session_state: st.session_state.line_flash_counter = 0

# ── Enhanced crossing events with real timestamps ──
if "crossing_events" not in st.session_state:
    st.session_state.crossing_events = []  # [(datetime_str, frame_no, track_id, direction)]
if "last_frame_rgb" not in st.session_state:
    st.session_state.last_frame_rgb = None  # Store last frame for snapshot
if "img_processed" not in st.session_state:
    st.session_state.img_processed = False

# ── Video replay support ──
if "video_temp_path" not in st.session_state:
    st.session_state.video_temp_path = None
if "video_completed" not in st.session_state:
    st.session_state.video_completed = False
if "_last_video_name" not in st.session_state:
    st.session_state._last_video_name = None

# ── Loading state ──
if "model_initializing" not in st.session_state:
    st.session_state.model_initializing = True

# ── Previous count states for pulse animation ──
if "prev_in_count" not in st.session_state: st.session_state.prev_in_count = -1
if "prev_out_count" not in st.session_state: st.session_state.prev_out_count = -1
if "prev_current_count" not in st.session_state: st.session_state.prev_current_count = -1
if "prev_unique_count" not in st.session_state: st.session_state.prev_unique_count = -1
if "prev_net_count" not in st.session_state: st.session_state.prev_net_count = -1

# ── Toast notification system ──
if "toast_messages" not in st.session_state:
    st.session_state.toast_messages = []
if "last_crossing_toast_time" not in st.session_state:
    st.session_state.last_crossing_toast_time = 0

# Live video capture state (main-thread approach — cap persists across reruns)
if "_video_cap" not in st.session_state:
    st.session_state._video_cap = None
if "_video_frame_idx" not in st.session_state:
    st.session_state._video_frame_idx = 0
if "_video_total_frames" not in st.session_state:
    st.session_state._video_total_frames = 0
if "_video_fps" not in st.session_state:
    st.session_state._video_fps = 30.0
if "_video_start_time" not in st.session_state:
    st.session_state._video_start_time = None

# ╔══════════════════════════════════════════════════════════════════╗
# ║          CORE FUNCTIONS (defined before sidebar handlers)        ║
# ╚══════════════════════════════════════════════════════════════════╝

def reset_counting():
    st.session_state.counted_ids = set()
    st.session_state.in_count = 0
    st.session_state.out_count = 0
    st.session_state.track_history = {}
    st.session_state.unique_ids = set()
    st.session_state.chart_timestamps = []
    st.session_state.chart_in_counts = []
    st.session_state.chart_out_counts = []
    st.session_state.chart_net_counts = []
    st.session_state.chart_current_counts = []
    st.session_state.peak_count = 0
    st.session_state.session_start_time = None
    st.session_state.crossing_events = []
    st.session_state.line_flash_counter = 0
    st.session_state.last_frame_rgb = None
    st.session_state.img_processed = False
    # Reset pulse animation tracking states
    st.session_state.prev_in_count = -1
    st.session_state.prev_out_count = -1
    st.session_state.prev_current_count = -1
    st.session_state.prev_unique_count = -1
    st.session_state.prev_net_count = -1
    # Clear toast messages and crossing toast throttle
    st.session_state.toast_messages = []
    st.session_state.last_crossing_toast_time = 0
    # Memory cleanup
    gc.collect()


def add_toast(message, title, toast_type="info"):
    """Add a toast notification to session state."""
    toast_id = f"{time.time()}_{random.randint(1000, 9999)}"
    st.session_state.toast_messages.append({
        "message": message,
        "title": title,
        "type": toast_type,
        "id": toast_id,
        "timestamp": time.time()
    })


def render_toasts():
    """Render all active toasts as HTML and clean up expired ones. Only render container div when active toasts exist."""
    now = time.time()
    active_toasts = []
    for toast in st.session_state.toast_messages:
        age = now - toast["timestamp"]
        if age < 3.0:
            active_toasts.append(toast)

    # Clean up expired toasts
    st.session_state.toast_messages = active_toasts

    if not active_toasts:
        return  # Don't render empty container

    toasts_html = ""
    for toast in active_toasts:
        time_str = datetime.now().strftime("%H:%M:%S")
        toasts_html += f"""<div class="toast-item {toast['type']}"><div class="toast-title">{toast['title']}</div><div class="toast-message">{toast['message']}</div><div class="toast-time">{time_str}</div></div>"""

    st.markdown(f"""<div class="toast-container">{toasts_html}</div>""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════╗
# ║          EXPORT FUNCTIONS                                      ║
# ╚══════════════════════════════════════════════════════════════════╝

def generate_csv_data():
    """Generate CSV data from crossing events and time-series."""
    # Build crossing events table
    event_rows = []
    for dt_str, frame_no, tid, direction in st.session_state.crossing_events:
        event_rows.append({
            "Timestamp": dt_str,
            "Frame": frame_no,
            "Track_ID": tid,
            "Event": direction,
            "Cumulative_In": None,
            "Cumulative_Out": None,
            "Net_Count": None
        })

    # Fill in cumulative counts at each event
    cumulative_in = 0
    cumulative_out = 0
    for row in event_rows:
        if row["Event"] == "ENTERED":
            cumulative_in += 1
        else:
            cumulative_out += 1
        row["Cumulative_In"] = cumulative_in
        row["Cumulative_Out"] = cumulative_out
        row["Net_Count"] = cumulative_in - cumulative_out

    df_events = pd.DataFrame(event_rows)

    # Build time-series summary
    ts_rows = []
    for i, frame in enumerate(st.session_state.chart_timestamps):
        ts_rows.append({
            "Frame": frame,
            "Entered": st.session_state.chart_in_counts[i],
            "Exited": st.session_state.chart_out_counts[i],
            "Net_Count": st.session_state.chart_net_counts[i],
            "In_Frame": st.session_state.chart_current_counts[i]
        })
    df_timeseries = pd.DataFrame(ts_rows)

    return df_events, df_timeseries


def export_to_csv():
    """Create a multi-sheet CSV string from all tracking data."""
    output = io.StringIO()

    # Header section
    output.write("=== ByteTrack Session Report ===\n")
    output.write(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    output.write(f"Total Entered: {st.session_state.in_count}\n")
    output.write(f"Total Exited: {st.session_state.out_count}\n")
    output.write(f"Net Count: {st.session_state.in_count - st.session_state.out_count}\n")
    output.write(f"Unique IDs: {len(st.session_state.unique_ids)}\n")
    output.write(f"Peak in Frame: {st.session_state.peak_count}\n")
    output.write(f"Confidence: {st.session_state.conf_threshold:.0%}\n")
    output.write(f"Line Position: {st.session_state.line_y_ratio:.0%}\n\n")

    # Crossing Events
    output.write("--- Crossing Events ---\n")
    df_events, df_timeseries = generate_csv_data()
    output.write(df_events.to_csv(index=False))
    output.write("\n")

    # Time-series data
    output.write("--- Frame-by-Frame Time Series ---\n")
    output.write(df_timeseries.to_csv(index=False))

    return output.getvalue()


def export_to_json():
    """Create a comprehensive JSON export of all tracking data."""
    df_events, df_timeseries = generate_csv_data()

    data = {
        "session_info": {
            "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_entered": st.session_state.in_count,
            "total_exited": st.session_state.out_count,
            "net_count": st.session_state.in_count - st.session_state.out_count,
            "unique_ids": len(st.session_state.unique_ids),
            "peak_in_frame": st.session_state.peak_count,
            "total_frames_processed": len(st.session_state.chart_timestamps),
            "confidence_threshold": st.session_state.conf_threshold,
            "counting_line_position": st.session_state.line_y_ratio,
        },
        "crossing_events": df_events.to_dict(orient="records"),
        "time_series": df_timeseries.to_dict(orient="records"),
    }

    return json.dumps(data, indent=2, default=str)


# Export buttons use direct download approach (lazy data generation on click)

# ╔══════════════════════════════════════════════════════════════════╗
# ║          SLIM SIDEBAR (4 sections) — FIXED LAYOUT     ║
# ╚══════════════════════════════════════════════════════════════════╝

# ── Section 1: Input Source ──
st.sidebar.markdown('<div class="sidebar-section"><h3><span class="section-dot dot-blue"></span>Input Source</h3></div>', unsafe_allow_html=True)
# Decouple selectbox key from source_select to allow welcome buttons to modify it
_choices = ["Choose...", "Upload Video", "Upload Image", "Use Webcam"]
if "source_select" not in st.session_state:
    st.session_state.source_select = "Choose..."
_current_idx = _choices.index(st.session_state.source_select)
_sel = st.sidebar.selectbox("Select Input Type", _choices, label_visibility="collapsed", index=_current_idx, key="_source_select_ui")
st.session_state.source_select = _sel
main_choice = st.session_state.source_select

# File uploaders placed immediately after selectbox
uploaded_file = None
uploaded_img = None
if main_choice == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Drag & Drop Video", type=["mp4", "avi", "mov"], key="video_uploader", label_visibility="collapsed")
elif main_choice == "Upload Image":
    uploaded_img = st.sidebar.file_uploader("Drag & Drop Image", type=["jpg", "jpeg", "png"], key="image_uploader", label_visibility="collapsed")

st.sidebar.markdown("---")

# ── Video replay: detect new upload or source change to invalidate cached temp file ──
if main_choice == "Upload Video" and uploaded_file:
    if st.session_state._last_video_name != uploaded_file.name:
        # New video uploaded — clean up old temp file and reset completion state
        if st.session_state.video_temp_path:
            try:
                os.unlink(st.session_state.video_temp_path)
            except OSError:
                pass
            st.session_state.video_temp_path = None
        st.session_state.video_completed = False
        st.session_state._last_video_name = uploaded_file.name
elif main_choice != "Upload Video":
    # User switched away from video input — clean up temp file
    if st.session_state.video_temp_path:
        try:
            os.unlink(st.session_state.video_temp_path)
        except OSError:
            pass
        st.session_state.video_temp_path = None
    st.session_state.video_completed = False
    st.session_state._last_video_name = None

# ── Section 2: Playback Controls ──
st.sidebar.markdown('<div class="sidebar-section"><h3><span class="section-dot dot-green"></span>Playback Controls</h3></div>', unsafe_allow_html=True)
col_start, col_pause, col_reset = st.sidebar.columns(3)
with col_start:
    start_btn = st.button("▶", use_container_width=True, help="Start tracking", key="btn_start")
with col_pause:
    pause_btn = st.button("⏸", use_container_width=True, help="Pause tracking", key="btn_pause")
with col_reset:
    reset_btn = st.button("⏹", use_container_width=True, help="Reset all counts", key="btn_reset")

if start_btn:
    st.session_state.is_running = True
    st.session_state.is_paused = False
    add_toast("Detection and tracking started", "Tracking Active", "info")
if pause_btn:
    st.session_state.is_paused = not st.session_state.is_paused
    if st.session_state.is_paused:
        add_toast("Tracking paused", "Paused", "warning")
if reset_btn:
    st.session_state.is_running = False
    st.session_state.is_paused = False
    st.session_state.video_completed = False
    # Release video capture if running
    if st.session_state._video_cap is not None:
        try:
            st.session_state._video_cap.release()
        except Exception:
            pass
        st.session_state._video_cap = None
    st.session_state._video_frame_idx = 0
    st.session_state._video_total_frames = 0
    if st.session_state.video_temp_path:
        try:
            os.unlink(st.session_state.video_temp_path)
        except OSError:
            pass
        st.session_state.video_temp_path = None
    reset_counting()
    add_toast("All counts and data reset", "Reset Complete", "danger")

# Speed Control — moved from Advanced Settings to Playback Controls
_speed_val = st.session_state.get("speed_slider", st.session_state.playback_speed)
st.sidebar.markdown(f'<div class="slider-label"><span>Playback Speed</span><span class="slider-val">{_speed_val:.2f}x</span></div>', unsafe_allow_html=True)
speed_slider = st.sidebar.slider("speed", min_value=0.25, max_value=3.0, value=st.session_state.playback_speed, step=0.25, key="speed_slider", label_visibility="collapsed")
st.session_state.playback_speed = speed_slider

st.sidebar.markdown("---")

# ── Section 3: Advanced Settings (collapsed expander) ──
with st.sidebar.expander("Advanced Settings", expanded=False):
    # Detection Settings
    # Show current value in slider labels
    # Read from widget key for freshest value (avoids 1-rerun label lag)
    _conf_val = st.session_state.get("conf_slider", st.session_state.conf_threshold)
    st.markdown(f'<div class="slider-label"><span>Confidence Threshold</span><span class="slider-val">{_conf_val:.0%}</span></div>', unsafe_allow_html=True)
    conf_slider = st.slider("conf", min_value=0.1, max_value=0.9, value=st.session_state.conf_threshold, step=0.05, key="conf_slider", label_visibility="collapsed")
    st.session_state.conf_threshold = conf_slider
    _line_val = st.session_state.get("line_slider", st.session_state.line_y_ratio)
    st.markdown(f'<div class="slider-label"><span>Counting Line Position</span><span class="slider-val">{_line_val:.0%}</span></div>', unsafe_allow_html=True)
    line_slider = st.slider("line", min_value=0.1, max_value=0.9, value=st.session_state.line_y_ratio, step=0.05, key="line_slider", label_visibility="collapsed")
    st.session_state.line_y_ratio = line_slider

    # Display Options
    st.session_state.show_boxes = st.checkbox("Bounding Boxes", value=st.session_state.show_boxes, key="toggle_boxes")
    st.session_state.show_ids = st.checkbox("Track IDs", value=st.session_state.show_ids, key="toggle_ids")
    st.session_state.show_counting_line = st.checkbox("Counting Line", value=st.session_state.show_counting_line, key="toggle_line")

st.sidebar.markdown("---")

# ── Section 4: Export & Snapshot ──
st.sidebar.markdown("""
<div class="sidebar-section">
    <h3><span class="section-dot dot-yellow"></span>Export &amp; Snapshot</h3>
</div>
""", unsafe_allow_html=True)

# Direct download buttons — always visible, generate data on click
# Lazy export: data generated only when download button is clicked, not on every rerun
def _get_csv_data():
    """Generate CSV data on-demand (not cached in session state to avoid stale data)."""
    return export_to_csv().encode('utf-8')

def _get_json_data():
    """Generate JSON data on-demand."""
    return export_to_json().encode('utf-8')

if st.session_state.crossing_events or st.session_state.chart_timestamps:
    st.sidebar.download_button(
        label="Download CSV", data=_get_csv_data(),
        file_name=f"bytetrack_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv", key="dl_csv", use_container_width=True
    )
    st.sidebar.download_button(
        label="Download JSON", data=_get_json_data(),
        file_name=f"bytetrack_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json", key="dl_json", use_container_width=True
    )
else:
    st.sidebar.download_button(
        label="Download CSV (empty)", data=b"No data yet",
        file_name="bytetrack_report_empty.csv", mime="text/csv",
        key="dl_csv", use_container_width=True, disabled=True
    )
    st.sidebar.download_button(
        label="Download JSON (empty)", data=b"{}",
        file_name="bytetrack_report_empty.json", mime="application/json",
        key="dl_json", use_container_width=True, disabled=True
    )

def _get_snapshot_bytes():
    """Generate snapshot PNG bytes on-demand (avoids cv2.imencode on every page rerun)."""
    if st.session_state.last_frame_rgb is not None:
        success, img_bytes = cv2.imencode('.png', st.session_state.last_frame_rgb)
        if success:
            return img_bytes.tobytes()
    return b""

if st.session_state.last_frame_rgb is not None:
    st.sidebar.download_button(
        label="Download Snapshot", data=_get_snapshot_bytes(),
        file_name=f"bytetrack_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
        mime="image/png", key="dl_snap", use_container_width=True
    )
else:
    st.sidebar.button("Snapshot (no frame)", use_container_width=True, key="btn_snap", disabled=True)

# ╔══════════════════════════════════════════════════════════════════╗
# ║                 CORE TRACKING LOGIC                              ║
# ╚══════════════════════════════════════════════════════════════════╝

@st.cache_resource
def load_model():
    """Load YOLO model using the working app.py path first, then sensible local fallbacks."""
    candidate_paths = [
        r"D:\Project data\Results\weights\best.pt",  # working path from app.py
        os.path.join(os.path.dirname(__file__), "best.pt"),
        os.path.join(os.path.dirname(__file__), "weights", "best.pt"),
        os.path.join(os.path.dirname(__file__), "Results", "weights", "best.pt"),
    ]

    for model_path in candidate_paths:
        if os.path.exists(model_path):
            return YOLO(model_path)

    raise FileNotFoundError(
        "Could not find YOLO weights file 'best.pt'. Checked: " + " | ".join(candidate_paths)
    )

# ╔══════════════════════════════════════════════════════════════════╗
# ║          LOADING SPINNER & PROGRESS BAR               ║
# ╚══════════════════════════════════════════════════════════════════╝

def render_loading_state(message="Initializing model", subtext="Loading YOLOv11 weights for the first time..."):
    """Render a centered loading spinner with message."""
    st.markdown(f"""
    <div class="loading-overlay">
        <div class="loading-spinner">
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
        </div>
        <div class="loading-text">{message}</div>
        <div class="loading-subtext">{subtext}</div>
    </div>
    """, unsafe_allow_html=True)


def render_progress_bar(current, total, elapsed=None, fps=None):
    """Render a custom dark-themed progress bar with timing info."""
    if total > 0:
        pct = min(100, (current / total) * 100)
    else:
        pct = 0

    # Calculate time estimates
    elapsed_str = "0.0s"
    eta_str = "--"
    if elapsed is not None and elapsed > 0:
        if elapsed >= 60:
            elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
        else:
            elapsed_str = f"{elapsed:.1f}s"
        if current > 0 and total > 0:
            eta = (elapsed / current) * (total - current)
            if eta >= 60:
                eta_str = f"~{int(eta // 60)}m {int(eta % 60)}s"
            else:
                eta_str = f"~{eta:.1f}s"

    st.markdown(f"""
    <div class="progress-container">
        <div class="progress-info">
            <span class="progress-label">Processing Frames</span>
            <span class="progress-pct">{pct:.0f}%</span>
        </div>
        <div class="progress-bar-track">
            <div class="progress-bar-fill" style="width: {pct:.1f}%;"></div>
        </div>
        <div class="progress-meta">
            <span>Frame {current} / {total}</span>
            <span>Elapsed: {elapsed_str}</span>
            <span>ETA: {eta_str}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════╗
# ║                  PROCESS FRAME                                   ║
# ╚══════════════════════════════════════════════════════════════════╝

def process_frame(frame):
    line_y_ratio = st.session_state.line_y_ratio
    fh, fw = frame.shape[:2]
    line_y = int(fh * line_y_ratio)

    model = load_model()
    results = model.track(frame, persist=True, conf=st.session_state.conf_threshold, verbose=False)
    current_ids = set()

    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            track_id = int(box.id) if box.id is not None else -1

            if track_id != -1:
                current_ids.add(track_id)
                st.session_state.unique_ids.add(track_id)
                center_y = int((y1 + y2) / 2)

                direction = None
                if track_id in st.session_state.track_history:
                    prev_y = st.session_state.track_history[track_id]
                    if prev_y < line_y and center_y >= line_y and track_id not in st.session_state.counted_ids:
                        st.session_state.in_count += 1
                        st.session_state.counted_ids.add(track_id)
                        direction = "in"
                        st.session_state.line_flash_counter = 10
                        # Log with real timestamp
                        st.session_state.crossing_events.append((
                            datetime.now().strftime("%H:%M:%S.%f")[:-3],
                            len(st.session_state.chart_timestamps),
                            track_id,
                            "ENTERED"
                        ))
                        # Crossing event toast (throttled — max 1 every 2 seconds)
                        now = time.time()
                        if now - st.session_state.last_crossing_toast_time >= 2.0:
                            st.session_state.last_crossing_toast_time = now
                            add_toast(f"Pedestrian {track_id} crossed — ENTERED", "Crossing Event", "event")
                    elif prev_y > line_y and center_y <= line_y and track_id not in st.session_state.counted_ids:
                        st.session_state.out_count += 1
                        st.session_state.counted_ids.add(track_id)
                        direction = "out"
                        st.session_state.line_flash_counter = 10
                        st.session_state.crossing_events.append((
                            datetime.now().strftime("%H:%M:%S.%f")[:-3],
                            len(st.session_state.chart_timestamps),
                            track_id,
                            "EXITED"
                        ))
                        # Crossing event toast (throttled — max 1 every 2 seconds)
                        now = time.time()
                        if now - st.session_state.last_crossing_toast_time >= 2.0:
                            st.session_state.last_crossing_toast_time = now
                            add_toast(f"Pedestrian {track_id} crossed — EXITED", "Crossing Event", "event")

                st.session_state.track_history[track_id] = center_y

                # Colored bounding boxes
                if st.session_state.show_boxes:
                    if direction == "in":
                        box_color = (0, 255, 0); id_color = (0, 255, 0)
                    elif direction == "out":
                        box_color = (0, 0, 255); id_color = (0, 0, 255)
                    else:
                        box_color = (200, 200, 200); id_color = (255, 0, 255)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)

                # ID labels
                if st.session_state.show_ids:
                    if direction == "in":
                        label = f"IN {track_id}"; id_color = (0, 255, 0)
                    elif direction == "out":
                        label = f"OUT {track_id}"; id_color = (0, 0, 255)
                    else:
                        label = f"ID {track_id}"; id_color = (255, 0, 255)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (int(x1), int(y1) - th - 8), (int(x1) + tw + 4, int(y1)), (0, 0, 0), -1)
                    cv2.putText(frame, label, (int(x1) + 2, int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, id_color, 2)

    # Counting line with flash
    if st.session_state.show_counting_line:
        if st.session_state.line_flash_counter > 0:
            fi = st.session_state.line_flash_counter / 10.0
            line_color = (0, int(255 * fi), int(255 * fi))
            line_thickness = 4
            st.session_state.line_flash_counter -= 1
        else:
            line_color = (0, 0, 255); line_thickness = 2
        dash_len = 30; gap_len = 15; x = 0
        while x < fw:
            end_x = min(x + dash_len, fw)
            cv2.line(frame, (x, line_y), (end_x, line_y), line_color, line_thickness, cv2.LINE_AA)
            x += dash_len + gap_len
        cv2.putText(frame, f"Counting Line ({line_y_ratio:.0%})", (fw - 260, line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, line_color, 2)

    current_count = len(current_ids)
    total_count = len(st.session_state.unique_ids)
    net_count = st.session_state.in_count - st.session_state.out_count

    if current_count > st.session_state.peak_count:
        st.session_state.peak_count = current_count

    # Stats overlay removed — all statistics are available in the Statistics tab and metric cards

    return frame, current_count, total_count

# ╔══════════════════════════════════════════════════════════════════╗
# ║              CHART & RENDER FUNCTIONS                           ║
# ╚══════════════════════════════════════════════════════════════════╝

def render_realtime_chart():
    ts = st.session_state.chart_timestamps
    if len(ts) < 2: return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=st.session_state.chart_in_counts, mode='lines+markers', name='Entered (In)', line=dict(color='#3fb950', width=2.5, shape='spline'), marker=dict(size=4, color='#3fb950'), fill='tozeroy', fillcolor='rgba(63, 185, 80, 0.08)'))
    fig.add_trace(go.Scatter(x=ts, y=st.session_state.chart_out_counts, mode='lines+markers', name='Exited (Out)', line=dict(color='#f85149', width=2.5, shape='spline'), marker=dict(size=4, color='#f85149'), fill='tozeroy', fillcolor='rgba(248, 81, 73, 0.08)'))
    fig.add_trace(go.Scatter(x=ts, y=st.session_state.chart_net_counts, mode='lines+markers', name='Net Count', line=dict(color='#58a6ff', width=2, dash='dash', shape='spline'), marker=dict(size=3, color='#58a6ff')))
    fig.update_layout(paper_bgcolor='#161b22', plot_bgcolor='#0d1117', font=dict(color='#c9d1d9', family='sans-serif', size=12), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=11, color='#8b949e'), bgcolor='rgba(0,0,0,0)'), margin=dict(l=40, r=20, t=10, b=40), xaxis=dict(gridcolor='#21262d', linecolor='#30363d', tickfont=dict(size=10, color='#8b949e'), title_text='Frame', title_font=dict(size=11, color='#8b949e')), yaxis=dict(gridcolor='#21262d', linecolor='#30363d', tickfont=dict(size=10, color='#8b949e'), title_text='Count', title_font=dict(size=11, color='#8b949e')), height=220)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#21262d')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#21262d')
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_current_count_chart():
    ts = st.session_state.chart_timestamps
    if len(ts) < 2: return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=st.session_state.chart_current_counts, mode='lines', name='In Frame', line=dict(color='#e3b341', width=2, shape='spline'), fill='tozeroy', fillcolor='rgba(227, 179, 65, 0.15)'))
    fig.add_hline(y=st.session_state.peak_count, line_dash="dot", line_color="#a371f7", annotation_text=f"Peak: {st.session_state.peak_count}", annotation_position="top left", annotation_font=dict(size=10, color="#a371f7"))
    fig.update_layout(paper_bgcolor='#161b22', plot_bgcolor='#0d1117', font=dict(color='#c9d1d9', family='sans-serif', size=12), margin=dict(l=40, r=20, t=10, b=40), xaxis=dict(gridcolor='#21262d', linecolor='#30363d', tickfont=dict(size=10, color='#8b949e'), title_text='Frame', title_font=dict(size=11, color='#8b949e')), yaxis=dict(gridcolor='#21262d', linecolor='#30363d', tickfont=dict(size=10, color='#8b949e'), title_text='Count', title_font=dict(size=11, color='#8b949e')), height=180, showlegend=False)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#21262d')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#21262d')
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_activity_log_table():
    """Render the full activity log as an HTML table."""
    if not st.session_state.crossing_events:
        st.markdown("""
        <div style="text-align: center; padding: 20px; color: #8b949e; font-size: 13px;">
            No crossing events recorded yet. Run tracking to generate events.
        </div>
        """, unsafe_allow_html=True)
        return

    # Build table HTML
    rows_html = ""
    for dt_str, frame_no, tid, direction in reversed(st.session_state.crossing_events):
        css_class = "event-entered" if direction == "ENTERED" else "event-exited"
        arrow = "&#9660;" if direction == "ENTERED" else "&#9650;"
        rows_html += f"""
        <tr>
            <td>{dt_str}</td>
            <td>{frame_no}</td>
            <td>{tid}</td>
            <td class="{css_class}">{arrow} {direction}</td>
        </tr>"""

    total_events = len(st.session_state.crossing_events)
    entered_count = sum(1 for _, _, _, d in st.session_state.crossing_events if d == "ENTERED")
    exited_count = total_events - entered_count

    st.markdown(f"""
    <div style="margin-bottom: 12px; display: flex; gap: 16px; color: #8b949e; font-size: 12px;">
        <span><strong style="color: #f0f6fc;">{total_events}</strong> Total Events</span>
        <span><strong style="color: #3fb950;">{entered_count}</strong> Entered</span>
        <span><strong style="color: #f85149;">{exited_count}</strong> Exited</span>
    </div>
    <div style="max-height: 300px; overflow-y: auto; border: 1px solid #30363d; border-radius: 10px;">
    <table class="activity-log-table">
        <thead>
            <tr>
                <th>Timestamp</th>
                <th>Frame</th>
                <th>Track ID</th>
                <th>Event</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)


def render_summary_panel(total_frames, video_fps):
    in_c = st.session_state.in_count
    out_c = st.session_state.out_count
    net_c = in_c - out_c
    unique = len(st.session_state.unique_ids)
    peak = st.session_state.peak_count
    avg_current = 0
    if st.session_state.chart_current_counts:
        avg_current = sum(st.session_state.chart_current_counts) / len(st.session_state.chart_current_counts)
    if video_fps and video_fps > 0 and total_frames > 0:
        ds = total_frames / video_fps
        duration_str = f"{int(ds // 60)}m {int(ds % 60)}s" if ds >= 60 else f"{ds:.1f}s"
    else:
        duration_str = "N/A"

    st.markdown(f"""
    <div class="summary-panel elevation-1">
        <h3><span class="section-dot dot-blue"></span>Session Summary</h3>
        <div class="summary-grid">
            <div class="summary-card"><div class="sum-icon" style="color: #3fb950;">&#9660;</div><div class="sum-value" style="color: #3fb950;">{in_c}</div><div class="sum-label">Total Entered</div></div>
            <div class="summary-card"><div class="sum-icon" style="color: #f85149;">&#9650;</div><div class="sum-value" style="color: #f85149;">{out_c}</div><div class="sum-label">Total Exited</div></div>
            <div class="summary-card"><div class="sum-icon" style="color: #58a6ff;">&#9670;</div><div class="sum-value" style="color: #58a6ff;">{net_c}</div><div class="sum-label">Net Count</div></div>
            <div class="summary-card"><div class="sum-icon" style="color: #a371f7;">&#9638;</div><div class="sum-value" style="color: #a371f7;">{unique}</div><div class="sum-label">Unique IDs</div></div>
            <div class="summary-card"><div class="sum-icon" style="color: #e3b341;">&#9650;</div><div class="sum-value" style="color: #e3b341;">{peak}</div><div class="sum-label">Peak in Frame</div></div>
            <div class="summary-card"><div class="sum-icon" style="color: #e3b341;">&#9660;</div><div class="sum-value" style="color: #e3b341;">{avg_current:.1f}</div><div class="sum-label">Avg in Frame</div></div>
            <div class="summary-card"><div class="sum-icon" style="color: #8b949e;">&#9646;</div><div class="sum-value" style="color: #8b949e;">{total_frames}</div><div class="sum-label">Frames</div></div>
            <div class="summary-card"><div class="sum-icon" style="color: #8b949e;">&#8986;</div><div class="sum-value" style="color: #8b949e;">{duration_str}</div><div class="sum-label">Duration</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Charts
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown('<div class="chart-area"><div class="area-label">In / Out / Net Over Time</div></div>', unsafe_allow_html=True)
        render_realtime_chart()
    with chart_col2:
        st.markdown('<div class="chart-area"><div class="area-label">Pedestrians in Frame Over Time</div></div>', unsafe_allow_html=True)
        render_current_count_chart()

    # ── Full Activity Log ──
    st.markdown('<div class="chart-area"><div class="area-label">Full Activity Log</div></div>', unsafe_allow_html=True)
    render_activity_log_table()

    # ── Export buttons in summary ──
    st.markdown('<div class="chart-area" style="margin-top: 16px;"><div class="area-label">Export Session Data</div></div>', unsafe_allow_html=True)
    exp_col1, exp_col2, exp_col3 = st.columns(3)
    with exp_col1:
        st.download_button("Download CSV", data=_get_csv_data(), file_name=f"bytetrack_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv", key="sum_csv", use_container_width=True)
    with exp_col2:
        st.download_button("Download JSON", data=_get_json_data(), file_name=f"bytetrack_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json", key="sum_json", use_container_width=True)
    with exp_col3:
        if st.session_state.last_frame_rgb is not None:
            st.download_button("Download Snapshot", data=_get_snapshot_bytes(), file_name=f"bytetrack_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", mime="image/png", key="sum_snap", use_container_width=True)


def render_metric_cards(in_c, out_c, current_c, unique_c, net_c):
    """Render metric cards with pulse animation on value change."""
    # Detect value changes for pulse animation
    pulse_in = " metric-value-pulse" if st.session_state.prev_in_count != -1 and in_c != st.session_state.prev_in_count else ""
    pulse_out = " metric-value-pulse" if st.session_state.prev_out_count != -1 and out_c != st.session_state.prev_out_count else ""
    pulse_net = " metric-value-pulse" if st.session_state.prev_net_count != -1 and net_c != st.session_state.prev_net_count else ""
    pulse_unique = " metric-value-pulse" if st.session_state.prev_unique_count != -1 and unique_c != st.session_state.prev_unique_count else ""
    pulse_current = " metric-value-pulse" if st.session_state.prev_current_count != -1 and current_c != st.session_state.prev_current_count else ""

    st.markdown(f"""
    <div class="metric-cards-row" role="status" aria-live="polite" aria-label="Tracking metrics">
        <div class="metric-card card-in" aria-label="Entered: {in_c}"><div class="metric-icon icon-enter">&#9660;</div><div class="metric-value{pulse_in}">{in_c}</div><div class="metric-label">Entered</div></div>
        <div class="metric-card card-out" aria-label="Exited: {out_c}"><div class="metric-icon icon-exit">&#9650;</div><div class="metric-value{pulse_out}">{out_c}</div><div class="metric-label">Exited</div></div>
        <div class="metric-card card-net" aria-label="Net Count: {net_c}"><div class="metric-icon icon-net">&#9670;</div><div class="metric-value{pulse_net}">{net_c}</div><div class="metric-label">Net Count</div></div>
        <div class="metric-card card-unique" aria-label="Unique IDs: {unique_c}"><div class="metric-icon icon-unique">&#9638;</div><div class="metric-value{pulse_unique}">{unique_c}</div><div class="metric-label">Unique IDs</div></div>
        <div class="metric-card card-current" aria-label="In Frame: {current_c}"><div class="metric-icon icon-current">&#9673;</div><div class="metric-value{pulse_current}">{current_c}</div><div class="metric-label">In Frame</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Update previous values to current
    st.session_state.prev_in_count = in_c
    st.session_state.prev_out_count = out_c
    st.session_state.prev_current_count = current_c
    st.session_state.prev_unique_count = unique_c
    st.session_state.prev_net_count = net_c


def render_welcome_metrics(entered, in_frame, unique):
    """Render 3 minimal metric cards for the welcome state."""
    st.markdown(f"""
    <div class="metric-cards-minimal">
        <div class="metric-card card-in"><div class="metric-icon icon-enter">&#9660;</div><div class="metric-value">{entered}</div><div class="metric-label">Entered</div></div>
        <div class="metric-card card-current"><div class="metric-icon icon-current">&#9673;</div><div class="metric-value">{in_frame}</div><div class="metric-label">In Frame</div></div>
        <div class="metric-card card-unique"><div class="metric-icon icon-unique">&#9638;</div><div class="metric-value">{unique}</div><div class="metric-label">Unique IDs</div></div>
    </div>
    """, unsafe_allow_html=True)


def render_stats_panel(in_c, out_c, current_c, unique_c, net_c):
    st.markdown('<div class="stats-panel elevation-1"><h3><span class="section-dot dot-blue"></span>Live Statistics</h3>', unsafe_allow_html=True)
    for icon_cls, icon_char, label, value, color in [
        ("icon-enter", "&#9660;", "Total Entered", in_c, "#3fb950"), ("icon-exit", "&#9650;", "Total Exited", out_c, "#f85149"),
        ("icon-net", "&#9670;", "Net Count", net_c, "#58a6ff"), ("icon-unique", "&#9638;", "Unique Pedestrians", unique_c, "#a371f7"),
        ("icon-current", "&#9673;", "Currently in Frame", current_c, "#e3b341"), ("icon-peak", "&#9650;", "Peak Count", st.session_state.peak_count, "#a371f7"),
        ("icon-conf", "&#8857;", "Confidence", f"{st.session_state.conf_threshold:.0%}", "#58a6ff"),
        ("icon-line", "&#9654;", "Line Position", f"{st.session_state.line_y_ratio:.0%}", "#8b949e"),
        ("icon-speed", "&#9889;", "Speed", f"{st.session_state.playback_speed:.1f}x", "#e3b341"),
        ("icon-event", "&#8635;", "Events", len(st.session_state.crossing_events), "#c9d1d9"),
    ]:
        st.markdown(f'<div class="stats-item"><span class="stats-label"><span class="icon {icon_cls}">{icon_char}</span> {label}</span><span class="stats-value" style="color: {color};">{value}</span></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_welcome():
    """Welcome state with minimal metrics and functional quick-action buttons."""
    st.markdown("""
    <div class="welcome-state" role="region" aria-label="Welcome">
        <div class="welcome-icon"><svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="#58a6ff" stroke-width="1.5" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><line x1="16.5" y1="16.5" x2="21" y2="21"/></svg></div>
        <h2>Welcome to ByteTrack</h2>
        <p>Select an input source from the sidebar to start pedestrian detection and tracking. 
        Use the controls to adjust detection settings, toggle display options, and control playback.
        Export tracking data as CSV or JSON, or capture snapshots at any time.</p>
    </div>
    """, unsafe_allow_html=True)

    # Functional quick-action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Video", key="wc_video", use_container_width=True):
            st.session_state.source_select = "Upload Video"
            st.rerun()
    with col2:
        if st.button("Image", key="wc_image", use_container_width=True):
            st.session_state.source_select = "Upload Image"
            st.rerun()
    with col3:
        if st.button("Webcam", key="wc_webcam", use_container_width=True):
            st.session_state.source_select = "Use Webcam"
            st.rerun()

    # Show live session metrics on welcome (not hardcoded zeros)
    _current_in_frame = st.session_state.chart_current_counts[-1] if st.session_state.chart_current_counts else 0
    render_welcome_metrics(st.session_state.in_count, _current_in_frame, len(st.session_state.unique_ids))


def render_controls_bar():
    sc = "#3fb950" if not st.session_state.is_paused else "#d29922"
    st_txt = "Running" if not st.session_state.is_paused else "Paused"
    if not st.session_state.is_running: sc = "#8b949e"; st_txt = "Ready"
    st.markdown(f"""
    <div class="controls-bar-inner" style="display: flex; align-items: center; justify-content: space-between; background-color: var(--color-bg-secondary); border: 1px solid var(--color-border-default); border-radius: var(--radius-sm); padding: 10px 16px; margin-bottom: 12px; gap: 12px;">
        <div style="display: flex; align-items: center; gap: 10px; flex-shrink: 0;">
            <span style="width: 10px; height: 10px; border-radius: 50%; background-color: {sc};"></span>
            <span style="color: var(--color-text-secondary); font-size: 13px; font-weight: 600;">{st_txt}</span>
        </div>
        <div class="controls-bar-stats" style="display: flex; gap: 14px; color: var(--color-text-muted); font-size: 12px;">
            <span>Conf: <strong style="color: var(--color-accent-blue);">{st.session_state.conf_threshold:.0%}</strong></span>
            <span>Line: <strong style="color: var(--color-accent-blue);">{st.session_state.line_y_ratio:.0%}</strong></span>
            <span>Speed: <strong style="color: var(--color-accent-yellow);">{st.session_state.playback_speed:.1f}x</strong></span>
            <span>Peak: <strong style="color: var(--color-accent-purple);">{st.session_state.peak_count}</strong></span>
            <span>Events: <strong style="color: var(--color-text-secondary);">{len(st.session_state.crossing_events)}</strong></span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════╗
# ║          MULTI-PAGE TAB NAVIGATION                    ║
# ╚══════════════════════════════════════════════════════════════════╝

tab1, tab2, tab3 = st.tabs(["Detection", "Analytics", "System"])

# ══════════════════════════════════════════════════════════════════
#  TAB 1: Detection
# ══════════════════════════════════════════════════════════════════

with tab1:
    if main_choice == "Choose...":
        render_welcome()

    elif main_choice == "Upload Video":
        # Determine video availability from temp file (replay) or fresh upload
        has_temp = (st.session_state.video_temp_path is not None and
                    os.path.exists(st.session_state.video_temp_path))

        # ── State 1: No video available at all ──
        if not uploaded_file and not has_temp:
            st.markdown("""
            <div class="welcome-state" style="padding: 40px 30px;">
                <div class="welcome-icon"><svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="#58a6ff" stroke-width="1.5" stroke-linecap="round"><polygon points="5 3 19 12 5 21 5 3"/><line x1="22" y1="3" x2="22" y2="21"/></svg></div>
                <h2>Upload a Video</h2>
                <p>Use the file uploader in the sidebar to upload a video file.
                Supported formats: <strong style="color: #58a6ff;">MP4, AVI, MOV</strong>.</p>
            </div>
            """, unsafe_allow_html=True)

        # ── State 2: Video completed — show final state with Replay button ──
        elif st.session_state.video_completed and not st.session_state.is_running:
            # Render final metric cards from preserved session state
            _final_unique = len(st.session_state.unique_ids)
            _final_net = st.session_state.in_count - st.session_state.out_count
            _final_in_frame = st.session_state.chart_current_counts[-1] if st.session_state.chart_current_counts else 0
            render_metric_cards(st.session_state.in_count, st.session_state.out_count,
                                _final_in_frame, _final_unique, _final_net)
            render_controls_bar()

            if st.session_state.last_frame_rgb is not None:
                video_col, stats_col = st.columns([3, 1])
                with video_col:
                    st.markdown('<div class="video-display-area"><div class="area-label"><span class="dot" style="background-color: #3fb950; animation: none;"></span> Final Frame</div></div>', unsafe_allow_html=True)
                    st.image(st.session_state.last_frame_rgb, channels="RGB", use_container_width=True)
                with stats_col:
                    render_stats_panel(st.session_state.in_count, st.session_state.out_count,
                                       0, _final_unique, _final_net)

            st.success(f"Video processing complete! {len(st.session_state.crossing_events)} crossing events detected.")

            # Replay button
            replay_col1, replay_col2 = st.columns([1, 1])
            with replay_col1:
                if st.button("Replay Video", use_container_width=True, key="btn_replay"):
                    # Release old capture and reset for replay
                    if st.session_state._video_cap is not None:
                        try:
                            st.session_state._video_cap.release()
                        except Exception:
                            pass
                        st.session_state._video_cap = None
                    st.session_state._video_frame_idx = 0
                    st.session_state.is_running = True
                    st.session_state.video_completed = False
                    reset_counting()
                    st.rerun()
            with replay_col2:
                if st.button("Reset & Upload New", use_container_width=True, key="btn_reset_new"):
                    st.session_state.video_completed = False
                    if st.session_state._video_cap is not None:
                        try:
                            st.session_state._video_cap.release()
                        except Exception:
                            pass
                        st.session_state._video_cap = None
                    st.session_state._video_frame_idx = 0
                    if st.session_state.video_temp_path:
                        try:
                            os.unlink(st.session_state.video_temp_path)
                        except OSError:
                            pass
                        st.session_state.video_temp_path = None
                    reset_counting()
                    st.rerun()

        # ── State 3: Video available, not running, not completed — Video Ready ──
        elif not st.session_state.is_running and not st.session_state.video_completed:
            if uploaded_file or has_temp:
                st.markdown("""
                <div class="welcome-state" style="padding: 40px 30px;">
                    <div class="welcome-icon"><svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="#3fb950" stroke-width="1.5" stroke-linecap="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg></div>
                    <h2>Video Ready</h2>
                    <p>Video file uploaded successfully. Press the <strong style="color: #3fb950;">Start</strong> button in the sidebar to begin pedestrian detection and tracking.</p>
                </div>
                """, unsafe_allow_html=True)

        # ── State 4: Running — LIVE VIDEO via main-thread auto-advance ──
        # process_frame() accesses st.session_state internally, so it MUST
        # run on the main Streamlit thread. We store cv2.VideoCapture in
        # session_state and process frames on each rerun.
        if st.session_state.is_running and (uploaded_file or has_temp):

            # ── 4A: Initialize video capture on first run ──
            if st.session_state._video_cap is None:
                video_path = None
                if has_temp:
                    video_path = st.session_state.video_temp_path
                elif uploaded_file:
                    tfile = tempfile.NamedTemporaryFile(delete=False)
                    video_path = tfile.name
                    tfile.write(uploaded_file.read())
                    tfile.close()
                    st.session_state.video_temp_path = video_path

                if video_path:
                    # Pre-load model before processing
                    if st.session_state.model_initializing:
                        loading_ph = st.empty()
                        with loading_ph.container():
                            render_loading_state("Loading Detection Model", "Initializing YOLOv11 + ByteTrack...")
                        load_model()
                        st.session_state.model_initializing = False
                        st.session_state.model_loaded = True
                        loading_ph.empty()
                        add_toast("YOLOv11 model loaded and ready", "Model Ready", "success")

                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        st.error("Could not open video file. Please check the file format and try again.")
                        st.session_state.is_running = False
                        st.rerun()
                        st.stop()
                    else:
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                        st.session_state._video_cap = cap
                        st.session_state._video_total_frames = total_frames
                        st.session_state._video_fps = video_fps
                        st.session_state._video_frame_idx = 0
                        st.session_state._video_start_time = time.time()
                        reset_counting()
                        st.session_state.is_paused = False
                        st.session_state.video_completed = False
                        st.session_state.session_start_time = time.time()

            # ── 4B: Process frames on main thread (safe for st.session_state) ──
            cap = st.session_state._video_cap
            if cap is not None and cap.isOpened():

                if not st.session_state.is_paused:
                    # Calculate frames to process this rerun (~100ms of video time)
                    speed = st.session_state.playback_speed
                    vfps = st.session_state._video_fps
                    frames_per_rerun = max(1, int(vfps * speed * 0.1))

                    frame_rgb = None
                    fc = st.session_state._video_frame_idx
                    total = st.session_state._video_total_frames

                    for _ in range(frames_per_rerun):
                        ret, frame = cap.read()
                        if not ret:
                            # Video finished
                            st.session_state.video_completed = True
                            st.session_state.is_running = False
                            break

                        fc += 1
                        # process_frame() is UNTOUCHED (Rule #1) — runs safely on main thread
                        frame, current_count, total_count = process_frame(frame)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        st.session_state._video_frame_idx = fc
                        st.session_state.last_frame_rgb = frame_rgb

                        # Update chart data (same as frame-by-frame did)
                        st.session_state.chart_timestamps.append(fc)
                        st.session_state.chart_in_counts.append(st.session_state.in_count)
                        st.session_state.chart_out_counts.append(st.session_state.out_count)
                        _net = st.session_state.in_count - st.session_state.out_count
                        st.session_state.chart_net_counts.append(_net)
                        st.session_state.chart_current_counts.append(current_count)
                        if current_count > st.session_state.peak_count:
                            st.session_state.peak_count = current_count

                    # Compute display metrics
                    elapsed = time.time() - st.session_state._video_start_time
                    inc = st.session_state.in_count
                    outc = st.session_state.out_count
                    net = inc - outc
                    unique = len(st.session_state.unique_ids)
                    cur = st.session_state.chart_current_counts[-1] if st.session_state.chart_current_counts else 0
                    actual_fps = fc / elapsed if elapsed > 0 else 0

                    # Render progress, metrics, controls
                    render_progress_bar(fc, total, elapsed=elapsed, fps=actual_fps)
                    render_metric_cards(inc, outc, cur, unique, net)
                    render_controls_bar()

                    # Layout: video + stats
                    video_col, stats_col = st.columns([3, 1])
                    with video_col:
                        if frame_rgb is not None:
                            st.image(frame_rgb, channels="RGB", use_container_width=True)
                        elif st.session_state.last_frame_rgb is not None:
                            st.image(st.session_state.last_frame_rgb, channels="RGB", use_container_width=True)

                        st.markdown('<div class="chart-area"><div class="area-label">Pedestrian Count Over Time</div></div>', unsafe_allow_html=True)
                        render_realtime_chart()

                    with stats_col:
                        render_stats_panel(inc, outc, cur, unique, net)

                    render_toasts()

                    # Check if video just completed
                    if st.session_state.video_completed:
                        # Finalize — show summary
                        cap.release()
                        st.session_state._video_cap = None
                        st.success(f"Video processing complete! Processed {fc} frames in {elapsed:.1f}s.")
                        add_toast(f"Processed {fc} frames successfully", "Video Complete", "info")
                        render_summary_panel(fc, vfps)
                        st.rerun()
                    else:
                        # Auto-advance: trigger next rerun for live playback
                        st.rerun()

                else:
                    # ── 4C: PAUSED — show last frame, don't advance ──
                    fc = st.session_state._video_frame_idx
                    total = st.session_state._video_total_frames
                    elapsed = time.time() - st.session_state._video_start_time if st.session_state._video_start_time else 0
                    inc = st.session_state.in_count
                    outc = st.session_state.out_count
                    net = inc - outc
                    unique = len(st.session_state.unique_ids)
                    cur = st.session_state.chart_current_counts[-1] if st.session_state.chart_current_counts else 0

                    render_progress_bar(fc, total, elapsed=elapsed)
                    render_metric_cards(inc, outc, cur, unique, net)
                    render_controls_bar()

                    video_col, stats_col = st.columns([3, 1])
                    with video_col:
                        if st.session_state.last_frame_rgb is not None:
                            st.markdown('<div class="video-display-area"><div class="area-label"><span class="dot paused"></span> Live Video Feed — Paused</div></div>', unsafe_allow_html=True)
                            st.image(st.session_state.last_frame_rgb, channels="RGB", use_container_width=True)
                        st.markdown('<div class="chart-area"><div class="area-label">Pedestrian Count Over Time</div></div>', unsafe_allow_html=True)
                        render_realtime_chart()
                    with stats_col:
                        render_stats_panel(inc, outc, cur, unique, net)

                    render_toasts()

    elif main_choice == "Upload Image":
        # file_uploader is in sidebar — use uploaded_img variable
        # Show guidance when no image uploaded yet
        if not uploaded_img:
            st.markdown("""
            <div class="welcome-state" style="padding: 40px 30px;">
                <div class="welcome-icon"><svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="#58a6ff" stroke-width="1.5" stroke-linecap="round"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg></div>
                <h2>Upload an Image</h2>
                <p>Upload an image file from the sidebar to run pedestrian detection.
                Supported formats: <strong style="color: #58a6ff;">JPG, JPEG, PNG</strong>.</p>
            </div>
            """, unsafe_allow_html=True)
        elif uploaded_img:
            # Detect new image upload and invalidate cached result
            _new_img_name = uploaded_img.name
            if st.session_state.get("_last_img_name") != _new_img_name:
                st.session_state._last_img_name = _new_img_name
                st.session_state.img_processed = False
            # Cache image result to avoid re-processing on every Streamlit rerun
            if "img_processed" not in st.session_state or not st.session_state.img_processed:
                reset_counting()
                try:
                    file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
                    _img_raw = cv2.imdecode(file_bytes, 1)
                    if _img_raw is None:
                        st.error("Could not decode image. The file may be corrupt or in an unsupported format.")
                        st.session_state.img_processed = True
                    else:
                        _img_result, _current_count, _total_count = process_frame(_img_raw)
                        st.session_state.last_frame_rgb = cv2.cvtColor(_img_result, cv2.COLOR_BGR2RGB)
                        st.session_state._img_counts = (_current_count, _total_count)
                        st.session_state.img_processed = True
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                    st.session_state.img_processed = True

            # Render cached image result (no re-processing)
            if st.session_state.last_frame_rgb is not None and "_img_counts" in st.session_state:
                _current_count, _total_count = st.session_state._img_counts
                render_metric_cards(st.session_state.in_count, st.session_state.out_count, _current_count, _total_count, st.session_state.in_count - st.session_state.out_count)
                img_col, stats_col = st.columns([3, 1])
                with img_col:
                    st.markdown('<div class="video-display-area"><div class="area-label"><span class="dot"></span> Detection Result</div></div>', unsafe_allow_html=True)
                    st.image(st.session_state.last_frame_rgb, channels="RGB", use_container_width=True)
                with stats_col:
                    render_stats_panel(st.session_state.in_count, st.session_state.out_count, _current_count, _total_count, st.session_state.in_count - st.session_state.out_count)

    elif main_choice == "Use Webcam":
        # Show guidance when not running; require Start button press
        if not st.session_state.is_running:
            st.markdown("""
            <div class="welcome-state" style="padding: 40px 30px;">
                <div class="welcome-icon"><svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="#58a6ff" stroke-width="1.5" stroke-linecap="round"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg></div>
                <h2>Webcam Ready</h2>
                <p>Press the <strong style="color: #3fb950;">Start</strong> button in the sidebar to begin live webcam pedestrian detection and tracking.
                Use <strong style="color: #d29922;">Reset</strong> to stop streaming.</p>
            </div>
            """, unsafe_allow_html=True)

        # Only start webcam if is_running is True
        if st.session_state.is_running:
            # Never auto-reset webcam data — user presses Reset explicitly
            st.session_state.session_start_time = time.time()

            # Show loading spinner on first model load
            if st.session_state.model_initializing:
                loading_ph = st.empty()
                with loading_ph.container():
                    render_loading_state("Loading Detection Model", "Initializing YOLOv11 + ByteTrack...")
                model = load_model()
                st.session_state.model_initializing = False
                st.session_state.model_loaded = True
                loading_ph.empty()
                # Toast after model loads successfully
                add_toast("YOLOv11 model loaded and ready", "Model Ready", "success")

            cap = cv2.VideoCapture(0)
            cam_progress_ph = st.empty()
            try:
                if not cap.isOpened():
                    st.error("Could not access webcam. Please ensure a camera is connected and no other application is using it.")
                    st.session_state.is_running = False
                else:
                    # Placeholders for metric cards and controls bar (updated every 10 frames in loop)
                    metric_cards_ph = st.empty()
                    controls_bar_ph = st.empty()
                    with metric_cards_ph.container():
                        render_metric_cards(st.session_state.in_count, st.session_state.out_count, 0, len(st.session_state.unique_ids), st.session_state.in_count - st.session_state.out_count)
                    with controls_bar_ph.container():
                        render_controls_bar()

                    video_col, stats_col = st.columns([3, 1])
                    with video_col:
                        st.markdown('<div class="video-display-area"><div class="area-label"><span class="dot"></span> Live Webcam Feed</div></div>', unsafe_allow_html=True)
                        chart_placeholder = st.empty()
                    with stats_col:
                        st.markdown('<div class="stats-col-spacer" style="height: 14px;"></div>', unsafe_allow_html=True)

                    frame_count = 0
                    while st.session_state.is_running:
                        ret, frame = cap.read()
                        if not ret: break
                        if st.session_state.is_paused: time.sleep(0.1); continue

                        frame, current_count, total_count = process_frame(frame)
                        frame_count += 1
                        st.session_state.last_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        st.session_state.chart_timestamps.append(frame_count)
                        st.session_state.chart_in_counts.append(st.session_state.in_count)
                        st.session_state.chart_out_counts.append(st.session_state.out_count)
                        st.session_state.chart_net_counts.append(st.session_state.in_count - st.session_state.out_count)
                        st.session_state.chart_current_counts.append(current_count)
                        if len(st.session_state.chart_timestamps) > 10000:
                            st.session_state.chart_timestamps = st.session_state.chart_timestamps[-10000:]
                            st.session_state.chart_in_counts = st.session_state.chart_in_counts[-10000:]
                            st.session_state.chart_out_counts = st.session_state.chart_out_counts[-10000:]
                            st.session_state.chart_net_counts = st.session_state.chart_net_counts[-10000:]
                            st.session_state.chart_current_counts = st.session_state.chart_current_counts[-10000:]

                        video_col.image(st.session_state.last_frame_rgb, channels="RGB", use_container_width=True)
                        if st.session_state.playback_speed > 0:
                            delay = max(1, int((1.0 / st.session_state.playback_speed) * 10))
                            time.sleep(delay * 0.01)

                        # Update placeholders every 10 frames to prevent UI stacking
                        if frame_count % 10 == 0:
                            with stats_col:
                                render_stats_panel(st.session_state.in_count, st.session_state.out_count, current_count, total_count, st.session_state.in_count - st.session_state.out_count)
                            with metric_cards_ph.container():
                                render_metric_cards(st.session_state.in_count, st.session_state.out_count, current_count, total_count, st.session_state.in_count - st.session_state.out_count)
                            with controls_bar_ph.container():
                                render_controls_bar()
                            with chart_placeholder.container():
                                st.markdown('<div class="chart-area"><div class="area-label">Pedestrian Count Over Time</div></div>', unsafe_allow_html=True)
                                render_realtime_chart()
                            render_toasts()

                            # Update webcam live counter (show frame count + elapsed, not progress %)
                            cam_elapsed = time.time() - st.session_state.session_start_time if st.session_state.session_start_time else 0
                            if cam_elapsed >= 60:
                                elapsed_str = f"{int(cam_elapsed // 60)}m {int(cam_elapsed % 60)}s"
                            else:
                                elapsed_str = f"{cam_elapsed:.1f}s"
                            with cam_progress_ph.container():
                                st.markdown(f"""
                                <div class="progress-container">
                                    <div class="progress-info">
                                        <span class="progress-label" style="color: #3fb950;">&#9679; LIVE</span>
                                        <span class="progress-pct" style="color: #3fb950;">{frame_count} frames</span>
                                    </div>
                                    <div class="progress-meta">
                                        <span>Elapsed: {elapsed_str}</span>
                                        <span>Speed: {st.session_state.playback_speed:.1f}x</span>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                    st.session_state.is_running = False
                    cam_progress_ph.empty()
            finally:
                cap.release()

# ══════════════════════════════════════════════════════════════════
#  TAB 2: Analytics
# ══════════════════════════════════════════════════════════════════

with tab2:
    has_data = (st.session_state.in_count > 0 or st.session_state.out_count > 0
                or len(st.session_state.crossing_events) > 0
                or len(st.session_state.chart_timestamps) > 0)

    if has_data:
        in_c = st.session_state.in_count
        out_c = st.session_state.out_count
        net_c = in_c - out_c
        unique = len(st.session_state.unique_ids)

        # ── Top row: 4 summary stat cards (responsive via CSS grid) ──
        st.markdown(f"""
        <div class="analytics-grid-4" style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 20px;">
            <div class="analytics-stat-card">
                <div class="stat-icon" style="color: #3fb950;">&#9660;</div>
                <div class="stat-value" style="color: #3fb950;">{in_c}</div>
                <div class="stat-label">In Count</div>
            </div>
            <div class="analytics-stat-card">
                <div class="stat-icon" style="color: #f85149;">&#9650;</div>
                <div class="stat-value" style="color: #f85149;">{out_c}</div>
                <div class="stat-label">Out Count</div>
            </div>
            <div class="analytics-stat-card">
                <div class="stat-icon" style="color: #58a6ff;">&#9670;</div>
                <div class="stat-value" style="color: #58a6ff;">{net_c}</div>
                <div class="stat-label">Net Count</div>
            </div>
            <div class="analytics-stat-card">
                <div class="stat-icon" style="color: #a371f7;">&#9638;</div>
                <div class="stat-value" style="color: #a371f7;">{unique}</div>
                <div class="stat-label">Unique IDs</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Two charts side by side ──
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown('<div class="chart-area"><div class="area-label">In / Out / Net Over Time</div></div>', unsafe_allow_html=True)
            render_realtime_chart()
        with chart_col2:
            st.markdown('<div class="chart-area"><div class="area-label">Pedestrians in Frame Over Time</div></div>', unsafe_allow_html=True)
            render_current_count_chart()

        # ── Activity log table ──
        st.markdown('<div class="chart-area"><div class="area-label">Full Activity Log</div></div>', unsafe_allow_html=True)
        render_activity_log_table()

        # ── Export buttons section ──
        st.markdown('<div class="chart-area" style="margin-top: 16px;"><div class="area-label">Export Session Data</div></div>', unsafe_allow_html=True)
        exp_col1, exp_col2, exp_col3 = st.columns(3)
        with exp_col1:
            st.download_button("Download CSV", data=_get_csv_data(), file_name=f"bytetrack_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv", key="analytics_csv", use_container_width=True)
        with exp_col2:
            st.download_button("Download JSON", data=_get_json_data(), file_name=f"bytetrack_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json", key="analytics_json", use_container_width=True)
        with exp_col3:
            if st.session_state.last_frame_rgb is not None:
                st.download_button("Download Snapshot", data=_get_snapshot_bytes(), file_name=f"bytetrack_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", mime="image/png", key="analytics_snap", use_container_width=True)
            else:
                st.markdown('<div style="color: #484f58; font-size: 13px; text-align: center; padding: 10px;">No snapshot available</div>', unsafe_allow_html=True)

    else:
        # Improved empty state with actionable guidance
        st.markdown("""
        <div class="analytics-empty">
            <div class="empty-icon"><svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#484f58" stroke-width="1" stroke-linecap="round"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/><line x1="15" y1="3" x2="15" y2="21"/></svg></div>
            <h3>No tracking data yet</h3>
            <p>Run pedestrian detection first to populate analytics, charts, and crossing event logs.<br/>
            <strong style="color: #58a6ff;">Step 1:</strong> Select an input source (Video / Image / Webcam) from the sidebar.<br/>
            <strong style="color: #58a6ff;">Step 2:</strong> Press the <strong style="color: #3fb950;">Start</strong> button to begin tracking.<br/>
            <strong style="color: #58a6ff;">Step 3:</strong> Return here to view detailed analytics and export reports.</p>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  TAB 3: System 
# ══════════════════════════════════════════════════════════════════

with tab3:

    # ── Section 1: Detection Configuration ──
    st.markdown("""
    <div class="settings-section">
        <h3><span class="section-dot dot-blue"></span>Detection Configuration</h3>
        <div class="settings-row">
            <div class="settings-label">Confidence Threshold<small>Minimum detection confidence for pedestrian bounding boxes</small></div>
            <div class="settings-value">{conf_val}</div>
        </div>
        <div class="settings-row">
            <div class="settings-label">Counting Line Position<small>Vertical position of the crossing detection line (0% = top, 100% = bottom)</small></div>
            <div class="settings-value">{line_val}</div>
        </div>
    </div>
    """.format(
        conf_val=f"{st.session_state.conf_threshold:.0%}",
        line_val=f"{st.session_state.line_y_ratio:.0%}"
    ), unsafe_allow_html=True)

    # ── Section 2: Display Options ──
    boxes_state = "on" if st.session_state.show_boxes else "off"
    ids_state = "on" if st.session_state.show_ids else "off"
    line_state = "on" if st.session_state.show_counting_line else "off"
    st.markdown(f"""
    <div class="settings-section">
        <h3><span class="section-dot dot-purple"></span>Display Options</h3>
        <div class="settings-row">
            <div class="settings-label">Bounding Boxes<small>Draw colored bounding boxes around detected pedestrians</small></div>
            <div class="settings-value {boxes_state}">{"● ON" if st.session_state.show_boxes else "○ OFF"}</div>
        </div>
        <div class="settings-row">
            <div class="settings-label">Track IDs<small>Display unique tracking ID labels above each pedestrian</small></div>
            <div class="settings-value {ids_state}">{"● ON" if st.session_state.show_ids else "○ OFF"}</div>
        </div>
        <div class="settings-row">
            <div class="settings-label">Counting Line<small>Draw the virtual counting line on the video frame</small></div>
            <div class="settings-value {line_state}">{"● ON" if st.session_state.show_counting_line else "○ OFF"}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Section 3: Model Information ──
    st.markdown("""
    <div class="settings-section">
        <h3><span class="section-dot dot-green"></span>Model Information</h3>
        <div class="settings-row">
            <div class="settings-label">Detector Architecture<small>Object detection model used for pedestrian detection</small></div>
            <div class="settings-value" style="color: #58a6ff;">YOLOv11</div>
        </div>
        <div class="settings-row">
            <div class="settings-label">Tracking Algorithm<small>Multi-object tracking algorithm for persistent IDs</small></div>
            <div class="settings-value" style="color: #a371f7;">ByteTrack</div>
        </div>
        <div class="settings-row">
            <div class="settings-label">Detection Class<small>Object class being tracked</small></div>
            <div class="settings-value">Pedestrian</div>
        </div>
        <div class="settings-row">
            <div class="settings-label">Model Weights<small>Path to trained YOLOv11 model weights</small></div>
            <div class="settings-value" style="font-size: 12px; color: #8b949e;">weights/best.pt</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Section 4: About ByteTrack ──
    st.markdown("""
    <div class="settings-section">
        <h3><span class="section-dot dot-gray"></span>About ByteTrack</h3>
        <div class="settings-row">
            <div class="settings-label">Description</div>
            <div class="settings-value" style="color: #c9d1d9; font-size: 12px; text-align: right; max-width: 400px;">ByteTrack uses YOLOv11 for pedestrian detection combined with ByteTrack algorithm for multi-object tracking. Designed for real-time pedestrian counting and crossing event detection.</div>
        </div>
        <div class="settings-row">
            <div class="settings-label">Version</div>
            <div class="settings-value" style="color: #58a6ff;">v2.0</div>
        </div>
        <div class="settings-row">
            <div class="settings-label">Phase</div>
            <div class="settings-value" style="color: #e3b341;">Final</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Section 5: System Information ──
    total_events = len(st.session_state.crossing_events)
    unique_ids_count = len(st.session_state.unique_ids)
    peak = st.session_state.peak_count
    total_frames = len(st.session_state.chart_timestamps)
    avg_current = 0
    if st.session_state.chart_current_counts:
        avg_current = sum(st.session_state.chart_current_counts) / len(st.session_state.chart_current_counts)

    st.markdown(f"""
    <div class="settings-section">
        <h3><span class="section-dot dot-yellow"></span>System Information</h3>
        <div class="settings-row">
            <div class="settings-label">Total Crossing Events<small>Combined entered and exited events this session</small></div>
            <div class="settings-value">{total_events}</div>
        </div>
        <div class="settings-row">
            <div class="settings-label">Unique Track IDs<small>Number of unique pedestrian IDs encountered</small></div>
            <div class="settings-value" style="color: #a371f7;">{unique_ids_count}</div>
        </div>
        <div class="settings-row">
            <div class="settings-label">Peak Count in Frame<small>Maximum pedestrians detected simultaneously in a single frame</small></div>
            <div class="settings-value" style="color: #e3b341;">{peak}</div>
        </div>
        <div class="settings-row">
            <div class="settings-label">Total Frames Processed<small>Total number of frames analyzed in this session</small></div>
            <div class="settings-value">{total_frames}</div>
        </div>
        <div class="settings-row">
            <div class="settings-label">Average in Frame<small>Average number of pedestrians per frame</small></div>
            <div class="settings-value">{avg_current:.1f}</div>
        </div>
        <div class="settings-row">
            <div class="settings-label">Playback Speed<small>Current video/webcam playback speed multiplier</small></div>
            <div class="settings-value" style="color: #e3b341;">{st.session_state.playback_speed:.2f}x</div>
        </div>
        <!-- Running Status shows 3 states: Running, Paused, Idle -->
        <div class="settings-row">
            <div class="settings-label">Running Status<small>Current state of the tracking system</small></div>
            <div class="settings-value {"on" if st.session_state.is_running and not st.session_state.is_paused else ("paused" if st.session_state.is_paused else "off")}">{"● Running" if st.session_state.is_running and not st.session_state.is_paused else ("● Paused" if st.session_state.is_paused else "○ Idle")}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════════════╗
# ║          RENDER TOASTS BEFORE FOOTER                  ║
# ╚══════════════════════════════════════════════════════════════════╝

render_toasts()

# ╔══════════════════════════════════════════════════════════════════╗
# ║                        FOOTER                                   ║
# ╚══════════════════════════════════════════════════════════════════╝

st.markdown("""
<div class="app-footer">
    ByteTrack v2.0 | YOLOv11 + ByteTrack
</div>
""", unsafe_allow_html=True)
