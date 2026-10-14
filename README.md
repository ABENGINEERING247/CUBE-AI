# CUBE AI — Custom Entity Training & .pt Model Generator Studio

## Run locally
1. Install Python 3.10 or 3.11.
2. Open a terminal in this folder.
3. Create and activate a virtual environment (recommended).
4. Run `pip install -r requirements.txt`.
5. Run `streamlit run app.py`.

## What it does
- Uploads a YOLO object-detection dataset ZIP containing `data.yaml`/`data.yml`.
- Trains an Ultralytics YOLO model and exposes `best.pt`/`last.pt` weights.
- Tests `.pt` models on uploaded images and videos.
- Offers browser webcam inference through `streamlit-webrtc`.

## Dataset
Use a YOLO detection dataset with matching image and label folders. See the Dataset Guide tab in the app for an example `data.yaml`.

## Notes
- Training may be slow or exceed memory limits on CPU or hosted free tiers. Start with a small dataset, small model, low batch size, and few epochs.
- Use only images and datasets you have permission to use.
- Live webcam access requires browser camera permission and a supported WebRTC network path.
- The app generates object-detection weights; it does not automatically create a dataset or guarantee model accuracy.
