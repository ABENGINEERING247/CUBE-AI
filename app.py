import io
import os
import re
import shutil
import tempfile
import zipfile
import threading
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
    WEBRTC_AVAILABLE = True
except Exception:
    WEBRTC_AVAILABLE = False
    VideoProcessorBase = object

APP_TITLE = "CUBE | Custom Entity Training Studio"
CREATOR = "Designed by Certified Generative and Agentic AI Application Developer Engr. Bilal Mehmood"
DEFAULT_MODEL = "yolo11n.pt"

st.set_page_config(page_title="CUBE AI", page_icon="🧊", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root { --cube-green:#16a34a; }
.stApp { background: linear-gradient(180deg,#f7fbff 0%,#ffffff 55%,#f4fff7 100%); }
.block-container { padding-top: 1.5rem; max-width: 1450px; }
.cube-hero { padding: 1.4rem 1.6rem; border-radius: 22px; color:#fff;
background:linear-gradient(120deg,#102a43,#14532d 72%,#16a34a); box-shadow:0 12px 35px #0f172a18; }
.cube-hero h1 { margin:0; font-size:2.5rem; letter-spacing:.04em; }
.cube-hero p { margin:.45rem 0 0; font-size:1.05rem; opacity:.94; }
.creator { text-align:center; color:#475569; font-size:.86rem; padding:1rem 0 .2rem; }
div[data-testid="stMetric"] { background:white; border:1px solid #e2e8f0; padding:14px; border-radius:14px; }
div.stButton > button { border-radius:11px; font-weight:650; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="cube-hero">
  <h1>🧊 CUBE AI</h1>
  <p><b>Custom Entity Training &amp; .pt Model Generator Studio</b></p>
  <p>Prepare a YOLO dataset · Train a custom detector · Export a .pt model · Test it on images, videos, or a live webcam</p>
</div>
""", unsafe_allow_html=True)
st.write("")

with st.sidebar:
    st.markdown("## 🧭 CUBE Control Panel")
    st.caption("Custom object detection workflow")
    st.info("Training requires a correctly labelled YOLO detection dataset. A GPU is recommended for larger datasets.")
    st.markdown("### 📦 Runtime")
    st.write("Ultralytics:", "✅ Ready" if YOLO else "❌ Not installed")
    st.write("Live webcam:", "✅ Ready" if WEBRTC_AVAILABLE else "⚠️ Install streamlit-webrtc")
    st.markdown("---")
    st.caption("Tip: use a small dataset and 5–20 epochs for your first test.")

if "trained_model_path" not in st.session_state:
    st.session_state.trained_model_path = ""
if "dataset_dir" not in st.session_state:
    st.session_state.dataset_dir = ""
if "train_log" not in st.session_state:
    st.session_state.train_log = ""

def safe_extract_zip(uploaded_file, destination):
    """Extract a ZIP while rejecting paths that escape the target directory."""
    root = Path(destination).resolve()
    with zipfile.ZipFile(uploaded_file) as zf:
        for member in zf.infolist():
            name = member.filename
            if not name or name.startswith("/") or "\\\\" in name:
                raise ValueError("The ZIP contains an unsafe path.")
            target = (root / name).resolve()
            if target != root and root not in target.parents:
                raise ValueError("The ZIP contains a path-traversal entry.")
        zf.extractall(root)

def find_data_yaml(root):
    candidates = list(Path(root).rglob("data.yaml")) + list(Path(root).rglob("data.yml"))
    return str(candidates[0]) if candidates else ""

def read_yaml_names(yaml_path):
    # Lightweight display parser; Ultralytics itself performs full YAML parsing.
    names = []
    try:
        import yaml
        data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
        value = data.get("names", []) if isinstance(data, dict) else []
        if isinstance(value, dict):
            names = [str(value[k]) for k in sorted(value, key=lambda x: int(x) if str(x).isdigit() else str(x))]
        elif isinstance(value, list):
            names = [str(x) for x in value]
    except Exception:
        pass
    return names

def valid_dataset_yaml(path):
    if not path or not os.path.isfile(path):
        return False, "Could not find data.yaml/data.yml in the ZIP."
    try:
        import yaml
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return False, "Dataset YAML must contain a mapping."
        if "train" not in data or "names" not in data:
            return False, "Dataset YAML must define at least 'train' and 'names'."
        if not any(k in data for k in ("val", "valid", "validation")):
            return False, "Dataset YAML should define a validation split (usually 'val')."
        return True, "Dataset YAML structure looks good."
    except Exception as exc:
        return False, f"Could not parse YAML: {exc}"

def get_model(path):
    if YOLO is None:
        raise RuntimeError("Ultralytics is unavailable. Install requirements.txt first.")
    return YOLO(path)

def render_detections(frame, model, conf=0.25):
    results = model.predict(source=frame, conf=float(conf), verbose=False)
    if results:
        return results[0].plot()
    return frame

tab_train, tab_test, tab_live, tab_help = st.tabs([
    "🧠 Train Custom Model", "🖼️ Image / Video Test", "📷 Live Webcam", "📘 Dataset Guide"
])

with tab_train:
    st.subheader("1. Add your labelled dataset")
    st.markdown("Upload a **ZIP** containing a YOLO-format dataset and its `data.yaml` (or `data.yml`).")
    dataset_zip = st.file_uploader("YOLO dataset ZIP", type=["zip"], key="dataset_zip")
    if dataset_zip is not None:
        if st.button("📂 Validate and prepare dataset", type="primary"):
            workdir = tempfile.mkdtemp(prefix="cube_dataset_")
            try:
                safe_extract_zip(dataset_zip, workdir)
                yaml_path = find_data_yaml(workdir)
                ok, message = valid_dataset_yaml(yaml_path)
                if ok:
                    st.session_state.dataset_dir = workdir
                    st.session_state.dataset_yaml = yaml_path
                    st.success(message)
                    st.write("Dataset YAML:", yaml_path)
                    names = read_yaml_names(yaml_path)
                    if names:
                        st.write("Classes:", ", ".join(names))
                    st.info("Expected common layout: images/train, images/val, labels/train, labels/val, and data.yaml.")
                else:
                    st.error(message)
                    shutil.rmtree(workdir, ignore_errors=True)
            except Exception as exc:
                shutil.rmtree(workdir, ignore_errors=True)
                st.error(f"Dataset preparation failed: {exc}")

    if st.session_state.get("dataset_yaml"):
        st.caption(f"Prepared dataset: {st.session_state.dataset_yaml}")
    st.markdown("---")
    st.subheader("2. Configure training")
    col1, col2, col3 = st.columns(3)
    with col1:
        base_model = st.selectbox("Base model", ["yolo11n.pt", "yolo11s.pt", "yolov8n.pt", "yolov8s.pt"], index=0)
        epochs = st.number_input("Epochs", min_value=1, max_value=300, value=30, step=5)
    with col2:
        imgsz = st.selectbox("Image size", [320, 416, 512, 640, 768], index=3)
        batch = st.selectbox("Batch size", [2, 4, 8, 16, 32], index=1)
    with col3:
        device = st.selectbox("Compute device", ["auto", "cpu", "0"], index=0, help="auto uses Ultralytics' automatic device selection; 0 requests the first CUDA GPU.")
        patience = st.number_input("Early-stopping patience", min_value=0, max_value=100, value=20, step=5)

    st.warning("Training can take a long time and may exceed memory or hosting limits. On Streamlit Community Cloud, GPU training is generally not available; train locally or in a GPU notebook for larger datasets.")
    train_clicked = st.button("🚀 Train and generate best.pt", type="primary", use_container_width=True)
    if train_clicked:
        yaml_path = st.session_state.get("dataset_yaml", "")
        ok, message = valid_dataset_yaml(yaml_path)
        if not ok:
            st.error("Prepare a valid dataset ZIP first. " + message)
        elif YOLO is None:
            st.error("Ultralytics is not installed. Add ultralytics to requirements.txt and redeploy.")
        else:
            try:
                with st.spinner("Loading base model and training… keep this tab open."):
                    model = get_model(base_model)
                    kwargs = dict(data=yaml_path, epochs=int(epochs), imgsz=int(imgsz),
                                  batch=int(batch), patience=int(patience), project="cube_runs",
                                  name="custom_entity", exist_ok=True, workers=2, verbose=True)
                    if device != "auto":
                        kwargs["device"] = device
                    result = model.train(**kwargs)
                    save_dir = Path(getattr(result, "save_dir", "cube_runs/custom_entity"))
                    best_path = save_dir / "weights" / "best.pt"
                    last_path = save_dir / "weights" / "last.pt"
                    if best_path.exists():
                        st.session_state.trained_model_path = str(best_path.resolve())
                        st.success(f"Training complete! Best model: {best_path}")
                    elif last_path.exists():
                        st.session_state.trained_model_path = str(last_path.resolve())
                        st.warning("best.pt was not found; using last.pt instead.")
                    else:
                        st.error("Training finished, but no weights file was found. Check the training logs.")
                    st.session_state.train_log = str(save_dir.resolve())
            except Exception as exc:
                st.error(f"Training failed: {exc}")
                st.exception(exc)

    if st.session_state.trained_model_path and os.path.isfile(st.session_state.trained_model_path):
        model_path = st.session_state.trained_model_path
        st.success(f"Ready to test: {Path(model_path).name}")
        with open(model_path, "rb") as f:
            model_bytes = f.read()
        st.download_button("⬇️ Download trained .pt model", data=model_bytes,
                           file_name=Path(model_path).name, mime="application/octet-stream",
                           use_container_width=True)
        st.caption(f"Training output folder: {st.session_state.train_log or 'not recorded'}")

with tab_test:
    st.subheader("Test any compatible YOLO .pt model")
    uploaded_model = st.file_uploader("Upload model weights (.pt)", type=["pt"], key="test_model")
    use_trained = st.checkbox("Use model trained in this session", value=True,
                              disabled=not bool(st.session_state.trained_model_path))
    selected_model = st.session_state.trained_model_path if use_trained and st.session_state.trained_model_path else ""
    if uploaded_model is not None:
        model_dir = Path(tempfile.gettempdir()) / "cube_uploaded_models"
        model_dir.mkdir(exist_ok=True)
        uploaded_path = model_dir / re.sub(r"[^A-Za-z0-9_.-]", "_", uploaded_model.name)
        uploaded_path.write_bytes(uploaded_model.getvalue())
        selected_model = str(uploaded_path)
    test_conf = st.slider("Detection confidence", 0.05, 0.95, 0.25, 0.05, key="test_conf")
    media = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "webp", "bmp", "mp4", "mov", "avi"], key="test_media")
    if media is not None:
        suffix = Path(media.name).suffix.lower()
        if suffix in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
            image = Image.open(io.BytesIO(media.getvalue())).convert("RGB")
            st.image(image, caption=f"Input: {media.name}", use_container_width=True)
            if st.button("🔍 Run image detection", type="primary"):
                if not selected_model:
                    st.error("Upload a .pt model or train one first.")
                elif YOLO is None:
                    st.error("Ultralytics is not installed.")
                else:
                    try:
                        with st.spinner("Detecting objects…"):
                            detector = get_model(selected_model)
                            arr = np.array(image)
                            result_img = render_detections(arr, detector, test_conf)
                        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Detection result", use_container_width=True)
                    except Exception as exc:
                        st.error(f"Inference failed: {exc}")
        else:
            st.info("Video inference runs frame-by-frame and may be slow on CPU.")
            if st.button("▶️ Process uploaded video", type="primary"):
                if not selected_model:
                    st.error("Upload a .pt model or train one first.")
                else:
                    temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                    temp_in.write(media.getvalue())
                    temp_in.close()
                    cap = cv2.VideoCapture(temp_in.name)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if not fps or fps <= 0 or fps > 120:
                        fps = 20.0
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
                    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    temp_out.close()
                    writer = cv2.VideoWriter(temp_out.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                    try:
                        detector = get_model(selected_model)
                        progress = st.progress(0.0)
                        status = st.empty()
                        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                        count = 0
                        while True:
                            ok, frame = cap.read()
                            if not ok:
                                break
                            out = render_detections(frame, detector, test_conf)
                            writer.write(out)
                            count += 1
                            if total > 0 and count % 5 == 0:
                                progress.progress(min(count / total, 1.0))
                                status.write(f"Processed {count}/{total} frames")
                        cap.release()
                        writer.release()
                        with open(temp_out.name, "rb") as f:
                            video_bytes = f.read()
                        st.video(video_bytes)
                        st.download_button("⬇️ Download annotated video", video_bytes,
                                           file_name="cube_annotated_video.mp4", mime="video/mp4")
                    except Exception as exc:
                        cap.release()
                        writer.release()
                        st.error(f"Video processing failed: {exc}")
                    finally:
                        try: os.unlink(temp_in.name)
                        except OSError: pass
                    try: os.unlink(temp_out.name)
                    except OSError: pass

with tab_live:
    st.subheader("Live real-time webcam detection")
    st.write("Choose a model and allow camera access in your browser. For public deployments, configure WebRTC/STUN/TURN if your network blocks peer connections.")
    live_uploaded_model = st.file_uploader("Upload .pt model for webcam", type=["pt"], key="live_model")
    live_use_trained = st.checkbox("Use session-trained model", value=True,
                                   disabled=not bool(st.session_state.trained_model_path), key="live_use_trained")
    live_model_path = st.session_state.trained_model_path if live_use_trained and st.session_state.trained_model_path else ""
    if live_uploaded_model is not None:
        model_dir = Path(tempfile.gettempdir()) / "cube_uploaded_models"
        model_dir.mkdir(exist_ok=True)
        live_path = model_dir / ("live_" + re.sub(r"[^A-Za-z0-9_.-]", "_", live_uploaded_model.name))
        live_path.write_bytes(live_uploaded_model.getvalue())
        live_model_path = str(live_path)
    live_conf = st.slider("Live confidence threshold", 0.05, 0.95, 0.35, 0.05, key="live_conf")
    if not WEBRTC_AVAILABLE:
        st.error("Live webcam dependencies are missing. Install streamlit-webrtc and av from requirements.txt, then restart.")
    elif YOLO is None:
        st.error("Ultralytics is not installed.")
    elif not live_model_path:
        st.info("Upload a .pt file above or train a model in the first tab to enable live detection.")
    else:
        class CUBEVideoProcessor(VideoProcessorBase):
            model = None
            confidence = 0.35
            lock = threading.Lock()

            def __init__(self):
                self.detector = None
                self.error = ""

            def recv(self, frame):
                image = frame.to_ndarray(format="bgr24")
                try:
                    with CUBEVideoProcessor.lock:
                        if self.detector is None:
                            self.detector = get_model(CUBEVideoProcessor.model)
                        output = render_detections(image, self.detector, CUBEVideoProcessor.confidence)
                    return type(frame).from_ndarray(output, format="bgr24")
                except Exception as exc:
                    self.error = str(exc)
                    cv2.putText(image, "CUBE inference error", (15, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    return type(frame).from_ndarray(image, format="bgr24")

        CUBEVideoProcessor.model = live_model_path
        CUBEVideoProcessor.confidence = live_conf
        webrtc_streamer(
            key="cube-live-webcam",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=CUBEVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

with tab_help:
    st.subheader("Dataset format guide")
    st.markdown("""
**Recommended ZIP structure**

```text
my_dataset/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

A minimal `data.yaml` example:

```yaml
path: /absolute/or/relative/dataset/root
train: images/train
val: images/val
names:
  0: person
  1: helmet
  2: safety_vest
```

- Each image needs a matching `.txt` label file in the corresponding labels folder.
- Each label row uses YOLO detection format: `class_id x_center y_center width height`; coordinates are normalized from 0 to 1.
- Class IDs start at `0` and must match the `names` list or dictionary.
- Include representative examples for every class, varied lighting/backgrounds, and a separate validation set.
- This studio trains **object detection** models. It does not create face recognition, image classification, segmentation, or pose models from arbitrary folders.
- A `.pt` file is a PyTorch checkpoint, not a standalone application. Keep a copy of your dataset and training configuration.
""")
    st.markdown("### Troubleshooting")
    st.markdown("""
- **Training fails:** validate `data.yaml` paths and labels; try a smaller image size, batch size, and fewer epochs.
- **No GPU:** choose `auto` or `cpu`; CPU training can be very slow.
- **Webcam does not open:** grant camera permission, use HTTPS/localhost, and check whether a firewall blocks WebRTC.
- **Model has no detections:** check that the model was trained for the target classes and lower the confidence threshold.
- **Deployment:** local webcam inference is easiest on a laptop/desktop. Hosted browser webcam access uses WebRTC and depends on network configuration.
""")

st.markdown("---")
st.markdown(f'<div class="creator">✨ {CREATOR}<br>🧊 CUBE AI · Custom Entity Training &amp; Model Testing Studio</div>', unsafe_allow_html=True)
