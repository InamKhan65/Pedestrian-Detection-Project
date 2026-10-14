import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

st.set_page_config(
    page_title="StoreFlow Vision",
    page_icon="🚶",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    :root {
        --bg: #07101e;
        --bg-2: #0b1426;
        --surface: rgba(12, 22, 40, 0.80);
        --surface-2: rgba(16, 28, 48, 0.72);
        --line: rgba(148, 163, 184, 0.14);
        --text: #ecf3fb;
        --muted: #94a8c6;
        --accent: #6ed1ff;
        --accent-soft: rgba(110, 209, 255, 0.14);
        --pink-soft: rgba(255, 112, 166, 0.14);
        --gold-soft: rgba(250, 204, 21, 0.12);
        --green-soft: rgba(52, 211, 153, 0.14);
    }

    .stApp {
        background:
            radial-gradient(circle at 12% 0%, rgba(37, 99, 235, 0.16), transparent 24%),
            radial-gradient(circle at 92% 8%, rgba(255, 112, 166, 0.11), transparent 18%),
            radial-gradient(circle at 75% 90%, rgba(52, 211, 153, 0.09), transparent 18%),
            linear-gradient(180deg, #06101c 0%, #091321 40%, #09111f 100%);
        color: var(--text);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(7, 16, 30, 0.98), rgba(9, 17, 31, 0.98));
        border-right: 1px solid rgba(148, 163, 184, 0.08);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--muted);
    }

    .sidebar-brand {
        padding: 0.15rem 0 0.85rem 0;
        margin-bottom: 0.35rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.10);
    }

    .sidebar-eyebrow {
        color: #86d9ff;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 0.72rem;
        margin-bottom: 0.2rem;
        font-weight: 700;
    }

    .sidebar-title {
        color: #f7fbff;
        font-size: 1.42rem;
        line-height: 1.1;
        font-weight: 800;
        margin: 0;
    }

    .sidebar-copy {
        color: var(--muted);
        font-size: 0.85rem;
        margin-top: 0.35rem;
        line-height: 1.45;
    }

    .side-label {
        color: #f2f7fd;
        font-size: 0.98rem;
        font-weight: 700;
        margin: 1rem 0 0.35rem 0;
    }

    .side-note {
        color: var(--muted);
        font-size: 0.79rem;
        margin-bottom: 0.3rem;
    }

    .topbar {
        position: relative;
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 1rem;
        padding: 1.0rem 1.12rem;
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 22px;
        background:
            linear-gradient(135deg, rgba(12, 22, 40, 0.86), rgba(8, 16, 30, 0.78));
        box-shadow: 0 14px 36px rgba(0, 0, 0, 0.18);
        overflow: hidden;
        margin-bottom: 0.9rem;
        transition: transform 180ms ease, box-shadow 180ms ease;
    }

    .topbar:hover {
        transform: translateY(-1px);
        box-shadow: 0 18px 44px rgba(0, 0, 0, 0.22);
    }

    .topbar::before {
        content: "";
        position: absolute;
        inset: 0 auto 0 0;
        width: 5px;
        background: linear-gradient(180deg, #67d4ff 0%, #7b8dff 52%, #ff79a8 100%);
        opacity: 0.95;
    }

    .topbar-title {
        font-size: 1.28rem;
        font-weight: 800;
        letter-spacing: 0.01em;
        margin: 0;
        color: #f7fbff;
    }

    .topbar-sub {
        color: #a8bad2;
        font-size: 0.9rem;
        margin-top: 0.28rem;
        max-width: 860px;
        line-height: 1.48;
    }

    .chip-row {
        display: flex;
        gap: 0.48rem;
        flex-wrap: wrap;
        justify-content: flex-end;
        align-items: center;
    }

    .chip {
        border: 1px solid rgba(148, 163, 184, 0.16);
        background: rgba(148, 163, 184, 0.06);
        color: #dce8f6;
        font-size: 0.79rem;
        padding: 0.38rem 0.68rem;
        border-radius: 999px;
        white-space: nowrap;
        backdrop-filter: blur(6px);
    }

    .kpi-card {
        position: relative;
        border: 1px solid rgba(148, 163, 184, 0.12);
        background: linear-gradient(180deg, rgba(12, 22, 40, 0.76), rgba(11, 18, 33, 0.68));
        border-radius: 18px;
        padding: 0.82rem 0.88rem 0.78rem 0.88rem;
        min-height: 94px;
        overflow: hidden;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.12);
        transition: transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
    }

    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(148, 163, 184, 0.22);
        box-shadow: 0 16px 34px rgba(0, 0, 0, 0.18);
    }

    .kpi-card::before {
        content: "";
        position: absolute;
        left: 0;
        right: 0;
        top: 0;
        height: 3px;
        background: var(--accent-strip, linear-gradient(90deg, #67d4ff, #7b8dff));
    }

    .kpi-card::after {
        content: "";
        position: absolute;
        right: -20px;
        top: -20px;
        width: 90px;
        height: 90px;
        border-radius: 999px;
        background: var(--accent-glow, rgba(110, 209, 255, 0.10));
        filter: blur(6px);
    }

    .kpi-label {
        color: #98acca;
        font-size: 0.71rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.40rem;
        font-weight: 700;
    }

    .kpi-value {
        font-size: 1.72rem;
        line-height: 1;
        font-weight: 800;
        color: #fbfdff;
        margin-bottom: 0.18rem;
    }

    .kpi-note {
        color: #afc0d7;
        font-size: 0.79rem;
    }

    .shell {
        border: 1px solid rgba(148, 163, 184, 0.12);
        background: linear-gradient(180deg, rgba(12, 22, 40, 0.76), rgba(11, 18, 33, 0.64));
        border-radius: 18px;
        padding: 0.95rem;
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.11);
    }

    .shell-title {
        font-size: 0.79rem;
        color: #86d9ff;
        margin-bottom: 0.62rem;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-weight: 700;
    }

    .frame-shell {
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 22px;
        overflow: hidden;
        background: rgba(5, 10, 20, 0.55);
        box-shadow: 0 18px 42px rgba(0, 0, 0, 0.18);
    }

    .empty-state {
        min-height: 500px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        border-radius: 22px;
        border: 1px dashed rgba(148, 163, 184, 0.18);
        background:
            radial-gradient(circle at top, rgba(110, 209, 255, 0.08), transparent 28%),
            linear-gradient(180deg, rgba(12, 22, 40, 0.42), rgba(12, 22, 40, 0.18));
        color: var(--muted);
        padding: 2.2rem;
    }

    .empty-state h3 {
        margin: 0 0 0.48rem 0;
        font-size: 1.18rem;
        color: var(--text);
    }

    .empty-steps {
        margin-top: 0.85rem;
        color: #a9bdd4;
        font-size: 0.86rem;
        line-height: 1.55;
    }

    .mini-list {
        display: grid;
        gap: 0.42rem;
    }

    .mini-row {
        display: flex;
        justify-content: space-between;
        gap: 0.75rem;
        padding: 0.48rem 0;
        border-bottom: 1px solid rgba(148, 163, 184, 0.08);
        font-size: 0.92rem;
    }

    .mini-row:last-child {
        border-bottom: none;
    }

    .mini-label {
        color: #8ea5c1;
    }

    .mini-value {
        color: #f8fbff;
        font-weight: 700;
        text-align: right;
    }

    .section-label {
        font-size: 0.87rem;
        color: #9cb2ce;
        margin-bottom: 0.55rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-weight: 700;
    }

    .stButton > button, .stDownloadButton > button {
        border-radius: 14px;
        min-height: 2.8rem;
        font-weight: 700;
        border: 1px solid rgba(148, 163, 184, 0.10);
        background: linear-gradient(90deg, #ff5e63 0%, #ff6a88 55%, #ff8a5b 100%);
        color: white;
        box-shadow: 0 10px 22px rgba(255, 95, 99, 0.22);
    }

    .stButton > button:hover {
        filter: brightness(1.04);
        box-shadow: 0 14px 28px rgba(255, 95, 99, 0.28);
    }

    div[data-testid="stExpander"] {
        border: 1px solid rgba(148, 163, 184, 0.10);
        border-radius: 14px;
        background: rgba(12, 22, 40, 0.42);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        background: rgba(148, 163, 184, 0.06);
        color: #b8c8dc;
        padding-left: 0.9rem;
        padding-right: 0.9rem;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(110, 209, 255, 0.14) !important;
        color: #f7fbff !important;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

MODEL_CANDIDATES = [
    Path(r"D:\Humayun\Results\weights\best.pt"),
    Path("best.pt"),
    Path("weights/best.pt"),
    Path("runs/detect/train/weights/best.pt"),
    Path("/mnt/data/best.pt"),
]


def init_state() -> None:
    defaults = {
        "track_states": {},
        "unique_ids": set(),
        "in_count": 0,
        "out_count": 0,
        "processed_frames": 0,
        "frame_index": 0,
        "last_runtime": None,
        "last_mode": "Idle",
        "last_source": "None",
        "last_resolution": "—",
        "last_model_path": None,
        "last_summary": None,
        "video_temp_path": None,
        "video_signature": None,
        "video_name": None,
        "video_total_frames": 0,
        "video_resolution": "—",
        "video_current_frame": 0,
        "video_playing": False,
        "video_elapsed_accum": 0.0,
        "video_run_started_at": None,
        "last_frame_rgb": None,
        "last_frame_stats": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def reset_tracking_state() -> None:
    st.session_state.track_states = {}
    st.session_state.unique_ids = set()
    st.session_state.in_count = 0
    st.session_state.out_count = 0
    st.session_state.processed_frames = 0
    st.session_state.frame_index = 0
    st.session_state.last_runtime = None
    st.session_state.last_summary = None
    st.session_state.last_frame_rgb = None
    st.session_state.last_frame_stats = None


def release_video_state(delete_file: bool = False) -> None:
    temp_path = st.session_state.video_temp_path
    if delete_file and temp_path and Path(temp_path).exists():
        try:
            os.unlink(temp_path)
        except OSError:
            pass
    st.session_state.video_temp_path = None
    st.session_state.video_signature = None
    st.session_state.video_name = None
    st.session_state.video_total_frames = 0
    st.session_state.video_resolution = "—"
    st.session_state.video_current_frame = 0
    st.session_state.video_playing = False
    st.session_state.video_elapsed_accum = 0.0
    st.session_state.video_run_started_at = None
    st.session_state.last_frame_rgb = None
    st.session_state.last_frame_stats = None


def prepare_video_source(uploaded_file) -> None:
    if uploaded_file is None:
        release_video_state(delete_file=False)
        return

    signature = f"{uploaded_file.name}:{getattr(uploaded_file, 'size', 0)}"
    if st.session_state.video_signature == signature and st.session_state.video_temp_path:
        return

    release_video_state(delete_file=True)
    reset_tracking_state()

    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_video:
        tmp_video.write(uploaded_file.getbuffer())
        temp_path = tmp_video.name

    cap = cv2.VideoCapture(temp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    resolution = f"{frame_width} × {frame_height}" if frame_width and frame_height else "Unknown"

    st.session_state.video_temp_path = temp_path
    st.session_state.video_signature = signature
    st.session_state.video_name = uploaded_file.name
    st.session_state.video_total_frames = total_frames
    st.session_state.video_resolution = resolution
    st.session_state.video_current_frame = 0
    st.session_state.video_playing = False
    st.session_state.video_elapsed_accum = 0.0
    st.session_state.video_run_started_at = None


def resolve_model_path(manual_path: str) -> Optional[str]:
    manual_path = manual_path.strip()
    if manual_path:
        candidate = Path(manual_path)
        if candidate.exists():
            return str(candidate)

    env_path = os.getenv("PED_MODEL_PATH", "").strip()
    if env_path and Path(env_path).exists():
        return env_path

    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return None


def safe_int(value: float) -> int:
    return int(round(float(value)))


def draw_overlay_box(
    frame: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    color: Tuple[int, int, int],
    alpha: float = 0.12,
) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, top_left, bottom_right, color, 2)


def draw_counting_line(frame: np.ndarray, line_y: int) -> None:
    dash_len = 18
    gap_len = 12
    x = 0
    width = frame.shape[1]
    line_color = (74, 190, 255)
    while x < width:
        end_x = min(x + dash_len, width)
        cv2.line(frame, (x, line_y), (end_x, line_y), line_color, 1, cv2.LINE_AA)
        x += dash_len + gap_len


COUNTER_COOLDOWN_FRAMES = 10
TRACK_STALE_FRAMES = 90


def classify_side(center_y: int, line_y: int, dead_band: int) -> str:
    if center_y < line_y - dead_band:
        return "above"
    if center_y > line_y + dead_band:
        return "below"
    return "neutral"


def prune_track_states(frame_index: int) -> None:
    stale_ids = [
        track_id
        for track_id, state in st.session_state.track_states.items()
        if frame_index - int(state.get("last_seen_frame", frame_index)) > TRACK_STALE_FRAMES
    ]
    for track_id in stale_ids:
        st.session_state.track_states.pop(track_id, None)


def process_frame(
    frame: np.ndarray,
    model: YOLO,
    conf_threshold: float,
    line_ratio: float,
    show_ids: bool,
    show_boxes: bool,
    show_line: bool,
    use_tracking: bool,
) -> Tuple[np.ndarray, Dict[str, int]]:
    output = frame.copy()
    frame_h, _ = output.shape[:2]
    line_y = int(frame_h * line_ratio)
    dead_band = max(10, int(frame_h * 0.025))
    current_ids = set()
    raw_detection_count = 0

    st.session_state.frame_index += 1
    frame_index = st.session_state.frame_index

    if use_tracking:
        results = model.track(
            frame,
            persist=True,
            conf=conf_threshold,
            verbose=False,
            tracker="bytetrack.yaml",
            classes=[0],
        )
    else:
        results = model.predict(
            frame,
            conf=conf_threshold,
            verbose=False,
            classes=[0],
        )

    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        raw_detection_count = len(boxes)
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1_i, y1_i, x2_i, y2_i = map(safe_int, [x1, y1, x2, y2])
            center_y = safe_int((y1 + y2) / 2)

            track_id = None
            if use_tracking and getattr(box, "id", None) is not None:
                try:
                    track_id = int(float(box.id.item()))
                except Exception:
                    track_id = None

            if show_boxes:
                draw_overlay_box(output, (x1_i, y1_i), (x2_i, y2_i), (56, 189, 248))

            if use_tracking and track_id is not None:
                current_ids.add(track_id)
                st.session_state.unique_ids.add(track_id)

                state = st.session_state.track_states.get(
                    track_id,
                    {
                        "stable_side": None,
                        "last_side": "neutral",
                        "last_seen_frame": frame_index,
                        "last_count_frame": -10_000,
                    },
                )
                current_side = classify_side(center_y, line_y, dead_band)
                prev_stable_side = state.get("stable_side")
                cooldown_ok = frame_index - int(state.get("last_count_frame", -10_000)) >= COUNTER_COOLDOWN_FRAMES

                if current_side != "neutral":
                    if prev_stable_side == "above" and current_side == "below" and cooldown_ok:
                        st.session_state.in_count += 1
                        state["last_count_frame"] = frame_index
                    elif prev_stable_side == "below" and current_side == "above" and cooldown_ok:
                        st.session_state.out_count += 1
                        state["last_count_frame"] = frame_index
                    state["stable_side"] = current_side

                state["last_side"] = current_side
                state["last_seen_frame"] = frame_index
                st.session_state.track_states[track_id] = state

                if show_ids:
                    cv2.putText(
                        output,
                        f"ID {track_id}",
                        (x1_i, max(20, y1_i - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (241, 245, 249),
                        2,
                        cv2.LINE_AA,
                    )

    prune_track_states(frame_index)

    current_count = len(current_ids) if use_tracking else raw_detection_count
    unique_count = len(st.session_state.unique_ids) if use_tracking else raw_detection_count

    if show_line:
        draw_counting_line(output, line_y)

    stats = {
        "current": current_count,
        "unique": unique_count,
        "in": st.session_state.in_count,
        "out": st.session_state.out_count,
        "net": st.session_state.in_count - st.session_state.out_count,
        "detections": raw_detection_count,
    }
    return output, stats


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "—"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rem = seconds % 60
    return f"{minutes}m {rem:.1f}s"


def render_topbar(mode: str, model_name: str, model_status: str) -> None:
    last_runtime = format_duration(st.session_state.last_runtime)
    st.markdown(
        f"""
        <div class="topbar">
            <div>
                <div class="topbar-title">StoreFlow Vision</div>
                <div class="topbar-sub">AI-powered pedestrian detection, identity tracking, and directional in/out flow analysis for Pakistani utility store surveillance using YOLOv11m and ByteTrack.</div>
            </div>
            <div class="chip-row">
                <div class="chip">Mode · {mode}</div>
                <div class="chip">Weights · {model_name}</div>
                <div class="chip">Engine · {model_status}</div>
                <div class="chip">Last run · {last_runtime}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_strip(placeholder, stats: Dict[str, int]) -> None:
    items = [
        ("In", stats["in"], "inbound crossings", "linear-gradient(90deg, #34d399, #22c55e)", "rgba(52, 211, 153, 0.12)"),
        ("Out", stats["out"], "outbound crossings", "linear-gradient(90deg, #fb7185, #f97316)", "rgba(251, 113, 133, 0.12)"),
        ("Net", stats["net"], "occupancy delta", "linear-gradient(90deg, #67d4ff, #7b8dff)", "rgba(103, 212, 255, 0.11)"),
        ("Current", stats["current"], "active tracks", "linear-gradient(90deg, #a78bfa, #7c3aed)", "rgba(167, 139, 250, 0.11)"),
        ("Unique", stats["unique"], "total IDs seen", "linear-gradient(90deg, #facc15, #fb7185)", "rgba(250, 204, 21, 0.10)"),
    ]
    with placeholder.container():
        cols = st.columns(5)
        for col, (label, value, note, strip, glow) in zip(cols, items):
            with col:
                st.markdown(
                    f"""
                    <div class="kpi-card" style="--accent-strip:{strip}; --accent-glow:{glow};">
                        <div class="kpi-label">{label}</div>
                        <div class="kpi-value">{value}</div>
                        <div class="kpi-note">{note}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_empty_state(placeholder, title: str = "Ready for thesis demo") -> None:
    placeholder.markdown(
        f"""
        <div class="empty-state">
            <div>
                <h3>{title}</h3>
                <div>Load a store video, image, or webcam feed from the left panel to launch live pedestrian detection and directional flow monitoring.</div>
                <div class="empty-steps">YOLOv11m detects pedestrians · ByteTrack keeps identities stable · the counting line drives in/out analytics.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_shell(placeholder, title: str, items: Dict[str, str]) -> None:
    rows = "".join(
        f'<div class="mini-row"><div class="mini-label">{k}</div><div class="mini-value">{v}</div></div>'
        for k, v in items.items()
    )
    placeholder.markdown(
        f"""
        <div class="shell">
            <div class="shell-title">{title}</div>
            <div class="mini-list">{rows}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def update_frame(placeholder, frame_rgb: np.ndarray) -> None:
    with placeholder.container():
        st.markdown('<div class="frame-shell">', unsafe_allow_html=True)
        st.image(frame_rgb, channels="RGB", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


def finalize_run(mode_label: str, source_label: str, resolution: str, elapsed: float, stats: Dict[str, int]) -> None:
    st.session_state.last_mode = mode_label
    st.session_state.last_source = source_label
    st.session_state.last_resolution = resolution
    st.session_state.last_runtime = elapsed
    st.session_state.last_summary = {**stats, "runtime": elapsed, "resolution": resolution}


# Sidebar
st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        <div class="sidebar-eyebrow">Thesis dashboard</div>
        <div class="sidebar-title">StoreFlow Vision</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown('<div class="side-label">Control source</div>', unsafe_allow_html=True)
model_override = st.sidebar.text_input(
    "Model path",
    value=r"D:\Humayun\Results\weights\best.pt",
    placeholder="best.pt or full path",
    label_visibility="visible",
)

st.sidebar.markdown('<div class="side-label">Input source</div>', unsafe_allow_html=True)
source_mode = st.sidebar.radio(
    "Choose input",
    ["Upload Video", "Upload Image", "Use Webcam"],
    index=0,
)

uploaded_video = None
process_video = False
uploaded_image = None
process_image = False
start_webcam = False
pause_video = False
stop_video = False

if source_mode == "Upload Video":
    uploaded_video = st.sidebar.file_uploader("Drag and drop video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video is not None:
        prepare_video_source(uploaded_video)
        controls = st.sidebar.columns(3)
        process_video = controls[0].button("Start", use_container_width=True, type="primary")
        pause_video = controls[1].button("Pause", use_container_width=True)
        stop_video = controls[2].button("Stop", use_container_width=True)
    else:
        release_video_state(delete_file=True)
elif source_mode == "Upload Image":
    uploaded_image = st.sidebar.file_uploader("Drag and drop image", type=["jpg", "jpeg", "png"])
    process_image = st.sidebar.button("Analyse image", use_container_width=True, type="primary")
elif source_mode == "Use Webcam":
    start_webcam = st.sidebar.button("Start webcam", use_container_width=True, type="primary")

webcam_limit = 1500
with st.sidebar.expander("Advanced settings", expanded=False):
    conf_threshold = st.slider("Confidence", 0.10, 0.90, 0.40, 0.01)
    line_ratio = st.slider("Line position", 0.20, 0.80, 0.50, 0.01)
    show_boxes = st.checkbox("Show boxes", value=True)
    show_ids = st.checkbox("Show labels", value=False)
    show_line = st.checkbox("Show counting line", value=True)
    webcam_limit = st.number_input(
        "Webcam frame limit",
        min_value=100,
        max_value=5000,
        value=1500,
        step=100,
    )
    if st.button("Reset counters", use_container_width=True):
        reset_tracking_state()
        st.success("Tracking state reset.")

resolved_model_path = resolve_model_path(model_override)
model = None
model_status = "Not found"
if resolved_model_path:
    try:
        model = load_model(resolved_model_path)
        st.session_state.last_model_path = resolved_model_path
        model_status = "Loaded"
    except Exception as exc:  # pragma: no cover
        st.error(f"Model loading failed: {exc}")
        model_status = "Load error"

chip_mode = source_mode.replace("Upload ", "")
chip_model = Path(resolved_model_path).name if resolved_model_path else "No weights"
render_topbar(chip_mode, chip_model, model_status)

stats_placeholder = st.empty()
current_stats = {
    "in": st.session_state.in_count,
    "out": st.session_state.out_count,
    "net": st.session_state.in_count - st.session_state.out_count,
    "current": 0,
    "unique": len(st.session_state.unique_ids),
}
render_kpi_strip(stats_placeholder, current_stats)
st.markdown("<div style='height: 0.52rem;'></div>", unsafe_allow_html=True)

main_col, side_col = st.columns([2.7, 0.95], gap="large")
with main_col:
    st.markdown('<div class="section-label">Live output</div>', unsafe_allow_html=True)
    frame_placeholder = st.empty()
    if st.session_state.last_frame_rgb is not None:
        update_frame(frame_placeholder, st.session_state.last_frame_rgb)
    else:
        render_empty_state(frame_placeholder, title="Ready to analyse")

with side_col:
    st.markdown('<div class="section-label">Overview</div>', unsafe_allow_html=True)
    status_placeholder = st.empty()
    st.markdown('<div style="height: 0.55rem;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Last run</div>', unsafe_allow_html=True)
    summary_placeholder = st.empty()
    st.markdown('<div style="height: 0.55rem;"></div>', unsafe_allow_html=True)
    progress_placeholder = st.empty()
    message_placeholder = st.empty()

render_info_shell(
    status_placeholder,
    "Overview",
    {
        "Processing mode": source_mode,
        "Model state": model_status,
        "Confidence": f"{conf_threshold:.2f}",
        "Line position": f"{line_ratio:.2f}",
        "Frames processed": str(st.session_state.processed_frames),
        "Tracked uniques": str(len(st.session_state.unique_ids)),
    },
)

last_summary = st.session_state.last_summary
render_info_shell(
    summary_placeholder,
    "Last run",
    {
        "Source": st.session_state.last_source,
        "Mode": st.session_state.last_mode,
        "Resolution": st.session_state.last_resolution,
        "Runtime": format_duration(st.session_state.last_runtime),
        "In / Out / Net": f"{st.session_state.in_count} / {st.session_state.out_count} / {st.session_state.in_count - st.session_state.out_count}",
        "Unique IDs": str(last_summary["unique"]) if last_summary else "—",
    },
)

progress_bar = progress_placeholder.progress(0)
message_placeholder.info("Standby")

if source_mode == "Upload Video":
    if process_video and st.session_state.video_temp_path:
        if st.session_state.video_current_frame == 0:
            reset_tracking_state()
        st.session_state.video_playing = True
        if st.session_state.video_run_started_at is None:
            st.session_state.video_run_started_at = time.perf_counter()
    if pause_video and st.session_state.video_playing:
        if st.session_state.video_run_started_at is not None:
            st.session_state.video_elapsed_accum += max(0.0, time.perf_counter() - st.session_state.video_run_started_at)
        st.session_state.video_playing = False
        st.session_state.video_run_started_at = None
    if stop_video and st.session_state.video_temp_path:
        if st.session_state.video_run_started_at is not None:
            st.session_state.video_elapsed_accum += max(0.0, time.perf_counter() - st.session_state.video_run_started_at)
        st.session_state.video_playing = False
        st.session_state.video_run_started_at = None
        st.session_state.video_current_frame = 0
        reset_tracking_state()
        message_placeholder.info("Video session reset. Press Start to run again.")


def refresh_side_panel(stats: Dict[str, int]) -> None:
    render_info_shell(
        status_placeholder,
        "Overview",
        {
            "Processing mode": source_mode,
            "Model state": model_status,
            "Confidence": f"{conf_threshold:.2f}",
            "Line position": f"{line_ratio:.2f}",
            "Frames processed": str(st.session_state.processed_frames),
            "Tracked uniques": str(stats["unique"]),
        },
    )


def update_ui(frame_rgb: np.ndarray, stats: Dict[str, int]) -> None:
    st.session_state.last_frame_rgb = frame_rgb
    st.session_state.last_frame_stats = stats
    update_frame(frame_placeholder, frame_rgb)
    render_kpi_strip(stats_placeholder, stats)
    refresh_side_panel(stats)


if model is None:
    message_placeholder.warning("No model file found. Use the left panel model path or place best.pt beside the app.")
else:
    if source_mode == "Upload Video":
        if uploaded_video is not None and st.session_state.video_temp_path:
            total_frames = max(1, int(st.session_state.video_total_frames or 0))
            progress_bar.progress(min(st.session_state.video_current_frame / total_frames, 1.0))

            if st.session_state.video_playing:
                cap = cv2.VideoCapture(st.session_state.video_temp_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.video_current_frame)
                ret, frame = cap.read()
                cap.release()

                if not ret:
                    elapsed = st.session_state.video_elapsed_accum
                    if st.session_state.video_run_started_at is not None:
                        elapsed += max(0.0, time.perf_counter() - st.session_state.video_run_started_at)
                    last_stats = {
                        "in": st.session_state.in_count,
                        "out": st.session_state.out_count,
                        "net": st.session_state.in_count - st.session_state.out_count,
                        "current": 0,
                        "unique": len(st.session_state.unique_ids),
                    }
                    finalize_run(
                        "Tracking + Counting",
                        st.session_state.video_name or "Uploaded video",
                        st.session_state.video_resolution,
                        elapsed,
                        last_stats,
                    )
                    render_info_shell(
                        summary_placeholder,
                        "Last run",
                        {
                            "Source": st.session_state.video_name or "Uploaded video",
                            "Mode": "Tracking + Counting",
                            "Resolution": st.session_state.video_resolution,
                            "Runtime": format_duration(elapsed),
                            "In / Out / Net": f"{last_stats['in']} / {last_stats['out']} / {last_stats['net']}",
                            "Unique IDs": str(last_stats["unique"]),
                        },
                    )
                    st.session_state.video_playing = False
                    st.session_state.video_run_started_at = None
                    message_placeholder.success("Video analysis complete.")
                else:
                    processed_frame, stats = process_frame(
                        frame=frame,
                        model=model,
                        conf_threshold=conf_threshold,
                        line_ratio=line_ratio,
                        show_ids=show_ids,
                        show_boxes=show_boxes,
                        show_line=show_line,
                        use_tracking=True,
                    )
                    st.session_state.processed_frames = st.session_state.video_current_frame + 1
                    st.session_state.video_current_frame += 1
                    rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    update_ui(rgb, stats)
                    progress_bar.progress(min(st.session_state.video_current_frame / total_frames, 1.0))
                    message_placeholder.info("Video playback running. Use Pause or Stop in the sidebar.")
                    time.sleep(0.03)
                    st.rerun()
            else:
                if st.session_state.video_current_frame > 0:
                    progress_bar.progress(min(st.session_state.video_current_frame / total_frames, 1.0))
                    message_placeholder.info("Video paused. Use Start in the sidebar to continue.")
                else:
                    message_placeholder.info("Video ready. Use Start in the sidebar.")

    elif source_mode == "Upload Image":

        if uploaded_image and process_image:
            reset_tracking_state()
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is None:
                message_placeholder.error("The uploaded image could not be decoded.")
            else:
                start_time = time.perf_counter()
                processed_frame, stats = process_frame(
                    frame=image,
                    model=model,
                    conf_threshold=conf_threshold,
                    line_ratio=line_ratio,
                    show_ids=show_ids,
                    show_boxes=show_boxes,
                    show_line=show_line,
                    use_tracking=False,
                )
                st.session_state.processed_frames = 1
                rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                update_ui(rgb, stats)
                elapsed = time.perf_counter() - start_time
                resolution = f"{image.shape[1]} × {image.shape[0]}"
                finalize_run("Single-frame detection", uploaded_image.name, resolution, elapsed, stats)
                render_info_shell(
                    summary_placeholder,
                    "Last run",
                    {
                        "Source": uploaded_image.name,
                        "Mode": "Single-frame detection",
                        "Resolution": resolution,
                        "Runtime": format_duration(elapsed),
                        "In / Out / Net": f"{stats['in']} / {stats['out']} / {stats['net']}",
                        "Unique IDs": str(stats["unique"]),
                    },
                )
                progress_bar.progress(1.0)
                message_placeholder.success("Image analysis complete.")

    elif source_mode == "Use Webcam":

        if start_webcam:
            reset_tracking_state()
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                message_placeholder.error("Webcam could not be opened on this machine.")
            else:
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                resolution = f"{frame_width} × {frame_height}" if frame_width and frame_height else "Unknown"

                start_time = time.perf_counter()
                last_stats = current_stats.copy()
                message_placeholder.info("Live webcam analysis running...")

                while cap.isOpened() and st.session_state.processed_frames < webcam_limit:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    processed_frame, stats = process_frame(
                        frame=frame,
                        model=model,
                        conf_threshold=conf_threshold,
                        line_ratio=line_ratio,
                        show_ids=show_ids,
                        show_boxes=show_boxes,
                        show_line=show_line,
                        use_tracking=True,
                    )
                    st.session_state.processed_frames += 1
                    last_stats = stats
                    rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    update_ui(rgb, stats)
                    progress_bar.progress(min(st.session_state.processed_frames / webcam_limit, 1.0))

                cap.release()
                elapsed = time.perf_counter() - start_time
                finalize_run("Live tracking + counting", "Local webcam", resolution, elapsed, last_stats)
                render_info_shell(
                    summary_placeholder,
                    "Last run",
                    {
                        "Source": "Local webcam",
                        "Mode": "Live tracking + counting",
                        "Resolution": resolution,
                        "Runtime": format_duration(elapsed),
                        "In / Out / Net": f"{last_stats['in']} / {last_stats['out']} / {last_stats['net']}",
                        "Unique IDs": str(last_stats["unique"]),
                    },
                )
                message_placeholder.success("Webcam session ended.")
