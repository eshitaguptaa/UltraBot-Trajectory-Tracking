# UltraBot Trajectory Tracking

AI-assisted breast ultrasound lesion detection, temporal tracking, and virtual biopsy planning.

The working baseline is **YOLOv8 + LSTM**. The Streamlit dashboard runs that existing pipeline. YOLOv13 is not connected.

Temporal tracking associates real YOLO box centers across frames with IoU and nearest-center distance. Missed frames keep the track ID for a short coast window and do **not** insert interpolated or sinusoidal positions. The LSTM is queried only after 20 consecutive real detections on the same track.

---

## Run the dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then in the browser:

1. Choose a sample video (`outputs/input.mp4`) or upload an ultrasound clip.
2. Set **Max frames** (start with 40 on CPU) and the YOLO confidence threshold.
3. Click **Run detection + tracking**.

The dashboard shows the processed HUD video, lesion coordinates and confidence, LSTM next-step prediction, trajectory, run-level evaluation, a 2D virtual biopsy plan, and a 2D robot simulation along the recommended path.

Weights are read-only: `models/best.pt` and `models/lstm_checkpoint.pth`.

### CLI demo (same pipeline)

```bash
python scripts/run_demo.py --max-frames 120
```

Output: `outputs/ultrabot_live.mp4`

---

## What the dashboard reports

Evaluation numbers are computed from the clip you just ran (detection coverage, YOLO confidence, consecutive-center displacement, LSTM one-step pixel error when a next detection exists).

Official validation mAP / precision / recall are **not** shown. No evaluation report is stored with `models/best.pt` in this workspace, and this project does not invent those scores.

Virtual biopsy is a **2D trajectory simulation** on a real ultrasound frame. It is not clinical or robotic control. A transparent weighted cost ranks candidate entry/target pairs; the lowest-cost feasible path is shown as the recommended trajectory. The Robot simulation tab animates a virtual needle along that recommended path. This is not a validated surgical planner.

---

## Project structure

```text
UltraBot-Trajectory-Tracking/
├── app.py                 # Streamlit dashboard
├── ultrabot/              # Shared pipeline used by the app and CLI
├── scripts/run_demo.py    # Command-line HUD demo
├── models/best.pt         # YOLOv8 weights (do not overwrite)
├── models/lstm_checkpoint.pth
├── outputs/input.mp4      # Sample ultrasound video
├── busv.ipynb             # Original notebook pipeline
├── data.yaml
└── requirements.txt
```

---

## Notebook (original)

```bash
jupyter notebook
```

Open `busv.ipynb`. Several cells still use older local paths (`runs/detect/train8/...`, `busi_video.avi`). Prefer the dashboard or `scripts/run_demo.py` for a working run.

---

## Author

Shivansh Arora
