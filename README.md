# Motion Detection on Home IP Cameras Using Edge AI

Open-source motion detection and classification system for home IP cameras
running on a Raspberry Pi 5, with a focus on minimizing false alarms.

This repository contains the prototype implementation that accompanies the
bachelor thesis _"Detekce a klasifikace pohybu na domácích IP kamerách
s využitím edge AI a filtrací falešných poplachů"_ (Unicorn University, 2026).

## Overview

The system processes a live RTSP stream from an IP camera through a
two-stage detection pipeline:

1. **MOG2 background subtraction** identifies regions of motion in the frame.
2. **YOLO classification** (YOLOv8 / YOLO11) classifies each motion region
   into person / vehicle / animal / other.
3. **Temporal tracker** stabilizes detections across frames and rejects
   single-frame false positives.
4. **Event recorder** saves an MP4 clip with pre- and post-event context
   only when an object has been confirmed by the tracker.

A separate **evaluator** records per-frame metrics (latency, CPU/RAM, FAR)
into CSV/JSON files for the experimental section of the thesis.

## Hardware

The reference deployment runs on:

- **Raspberry Pi 5** (8 GB RAM)
- **Reolink P340** IP camera (sub-stream: 896x512 H.264, main-stream: 4512x2512 H.265)
- Optional: **Raspberry Pi AI HAT+** with the Hailo-8 accelerator (26 TOPS) -
  not yet integrated, planned as future work in the thesis.

The code does not assume a specific camera model; any RTSP-capable camera
should work. The `reolink_collector.py` module is Reolink-specific and is
only needed for the comparative experiment in section 4.5 of the thesis.

## Installation

```bash
# Clone the repository
git clone https://github.com/Muth008/motion-detection.git
cd motion-detection

# Install dependencies
pip install -r requirements.txt

# Configure camera credentials
cp .env.example .env
# edit .env and set CAMERA_HOST, CAMERA_USERNAME, CAMERA_PASSWORD
```

The first run will download the YOLO model weights automatically (the model
file is configured in `config.py`, default is `yolo11n.pt`).

### Raspberry Pi specifics

OpenCV and ultralytics work out of the box on Raspberry Pi OS Bookworm 64-bit.
For headless operation set `SHOW_MAIN_WINDOW = False` and
`SHOW_DEBUG_WINDOWS = False` in `config.py` (already the default).

## Usage

### Live operation

```bash
python main.py
```

The system connects to the configured RTSP stream and starts the detection
pipeline. Recorded events are saved to `./recordings/` as MP4 + JPG thumbnail
+ JSON metadata.

### Experimental mode (offline evaluation)

For reproducible experiments the system can replay a video file instead of
a live stream:

```bash
# 1. Record a test video from the camera (5 minutes by default)
python record_video.py --duration 300

# 2. Annotate ground-truth events
python annotate_video.py test_video_<ts>_sub.mp4

# 3. Set INPUT_SOURCE in config.py to the recorded file, then run main.py
python main.py

# 4. Evaluate the session against the ground truth
python evaluate_experiment.py ground_truth_<video>.csv --results eval_results/
```

### Comparison with the camera's built-in AI (Reolink only)

To compare the system with the Reolink P340 built-in AI, run both
collectors in parallel:

```bash
# Terminal 1
python main.py

# Terminal 2 (start once main.py says "Sync file ready")
python reolink_collector.py
```

Both processes share a `session_sync.json` file so their outputs can be
aligned in time.

## Configuration

All parameters are in `config.py`. The main groups:

- **Camera**: RTSP stream selection (sub vs main), credentials from `.env`.
- **MOG2**: history, variance threshold, shadow detection.
- **Morphology**: opening / closing kernel size and iterations.
- **Contour filter**: min/max area, dimensions, aspect ratio.
- **YOLO**: model file, confidence threshold, IoU threshold,
  detection mode (`motion_regions` vs `full_frame`).
- **Tracker**: min hits to confirm, max age before LOST, IoU threshold.
- **Recorder**: pre/post buffer length, codec, output directory.
- **Evaluator**: session label, output directory.

The thesis describes seven experimental configurations - see the comments
in `config.py` for the exact parameter values used in each.

## Repository structure

```
motion-detection/
├── main.py                  Main orchestration loop
├── config.py                Centralized configuration (.env-aware)
├── camera.py                RTSP / file video reader
├── motion_detector.py       MOG2 background subtraction
├── object_detector.py       YOLO wrapper (Ultralytics)
├── tracker.py               Temporal tracker (state machine)
├── recorder.py              Event-triggered recording with circular buffer
├── evaluator.py             Per-frame metrics + CSV/JSON export
├── reolink_collector.py     Reolink HTTPS AI event collector (Section 4.5)
├── record_video.py          Standalone test-video recording utility
├── annotate_video.py        Ground-truth annotation tool
├── evaluate_experiment.py   Compare results to ground truth (TP/FP/FN/FAR)
├── evaluator_data/          Raw data from all 37 thesis experiments
│   ├── summary.csv          Aggregated results, one row per session
│   ├── sessions/            Per-frame CSV + JSON for each session (74 files)
│   ├── ground_truth/        Manual annotations of test videos
│   └── results/             TP/FP/FN evaluation outputs
├── requirements.txt         Python dependencies
├── .env.example             Template for camera credentials
├── .gitignore
└── LICENSE                  MIT
```

The `evaluator_data/` directory provides the full reproducibility material
behind the experimental section of the thesis - see its own README for
column descriptions and the experiment naming scheme.

## Experimental results (thesis summary)

The thesis evaluates seven experiments on a 14-event ground-truth dataset:

| Experiment | Setting compared           | Key finding |
|------------|----------------------------|-------------|
| Exp 1      | YOLO model (n / 11s / v8n) | All three models share the same recall - the bottleneck is upstream of YOLO. |
| Exp 2      | YOLO confidence threshold  | F1 stable from 0.25 to 0.55. |
| Exp 3      | Detection mode + imgsz     | `motion_regions` matches full-frame at 1/3 the latency. |
| Exp 4      | MOG2 sensitivity           | Higher sensitivity recovers far-distance recall but adds FPs. |
| Exp 5      | Best sub-stream config     | F1 = 42 % on sub-stream (896x512) - resolution-limited. |
| Exp 6      | Reference: main-stream     | F1 = 93 % on main-stream (4512x2512) offline - confirms architecture. |
| Exp 7      | FAR vs Reolink built-in AI | The system produced 0 false alarms across 9 scenarios; the Reolink AI failed in all 8 person-free scenarios. |

The headline finding: on a low-resolution sub-stream the detection ceiling is
set by the input resolution and the ability of MOG2 to generate contours for
distant objects, **not** by the YOLO classifier. The proposed direction for
future work is integration of a Hailo-8 hardware accelerator to enable
real-time main-stream processing.

## Citation

If you use this code or its results in your work, please cite the thesis:

> Šindelář, J. (2026). _Detekce a klasifikace pohybu na domácích IP kamerách
> s využitím edge AI a filtrací falešných poplachů_. Bachelor thesis,
> Unicorn University, Prague.

## License

MIT License - see [LICENSE](LICENSE) for details.
