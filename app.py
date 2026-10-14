import streamlit as st
import cv2
import numpy as np
import tempfile
import torch
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker

# ✅ Fix NumPy float issue
if not hasattr(np, 'float'):
    np.float = float

# ✅ Load YOLOv11 model
MODEL_PATH = r"D:\Project data\Results\weights\best.pt"
model = YOLO(MODEL_PATH)

# ✅ ByteTrack args
class Args:
    track_thresh = 0.5
    track_buffer = 30
    match_thresh = 0.8
    aspect_ratio_thresh = 1.6
    min_box_area = 10
    mot20 = False

tracker = BYTETracker(Args())
unique_ids = set()

# ✅ Process frame
def process_frame(frame):
    global unique_ids
    h, w = frame.shape[:2]

    results = model.predict(frame, conf=0.4, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            score = float(box.conf[0])
            cls = int(box.cls[0])
            if cls == 0:
                detections.append([x1, y1, x2, y2, score])

    detections = np.array(detections)

    # ✅ Handle empty detections to avoid IndexError
    if detections.shape[0] == 0:
        return frame, 0, len(unique_ids)

    online_targets = tracker.update(detections, [h, w], (h, w))

    current_ids = set()
    for t in online_targets:
        tlwh = t.tlwh
        track_id = t.track_id
        current_ids.add(track_id)
        unique_ids.add(track_id)

        x, y, w_box, h_box = map(int, tlwh)
        # ✅ Green bounding box
        cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
        # ✅ Dark Red ID text
        cv2.putText(frame, f"ID {track_id}", (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)  # Deep Pink


    current_count = len(current_ids)
    total_count = len(unique_ids)

    # ✅ Red bold text for stats (no background)
    cv2.putText(frame, f"Pedestrians: {current_count} | Total Unique: {total_count}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    return frame, current_count, total_count

# ✅ Streamlit UI
st.set_page_config(page_title="Pedestrian Tracking", layout="wide")

# ✅ Custom sidebar style
st.markdown("""
    <style>
    .sidebar .sidebar-content {
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("📌 Input Options")
main_choice = st.sidebar.selectbox("Select Input Type", ["Choose...", "Upload Video", "Upload Image", "Use Webcam"])

st.title("🚶 Pedestrian Detection & Tracking")
st.markdown("### Upload video/image or use webcam for real-time tracking.")

# ✅ Handle input choices
if main_choice == "Upload Video":
    uploaded_file = st.file_uploader("🎥 Drag & Drop Video", type=["mp4", "avi", "mov"])
    if uploaded_file:
        unique_ids.clear()
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame, _, _ = process_frame(frame)
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        cap.release()

elif main_choice == "Upload Image":
    uploaded_img = st.file_uploader("🖼️ Drag & Drop Image", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        unique_ids.clear()
        file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img, _, _ = process_frame(img)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB")

elif main_choice == "Use Webcam":
    st.info("🎦 Using webcam. Close the tab to stop streaming.")
    unique_ids.clear()
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame, _, _ = process_frame(frame)
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
    cap.release()
