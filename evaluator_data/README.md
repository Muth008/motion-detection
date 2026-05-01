# Motion Detection on Home IP Cameras Using Edge AI

Open-source motion detection and object classification prototype for home IP cameras running on a Raspberry Pi 5, with a focus on reducing false alarms in practical surveillance scenarios.

This repository accompanies the bachelor thesis:

> Šindelář, J. (2026). *Detekce a klasifikace pohybu na domácích IP kamerách s využitím edge AI a filtrací falešných poplachů*. Bachelor thesis, Unicorn University, Prague.

English thesis title:

> *Motion Detection and Classification on Home IP Cameras Using Edge AI with False Alarm Filtering*

---

## Overview

The system processes a live or recorded RTSP video stream through a multi-stage edge AI pipeline:

1. **Motion detection** – MOG2 background subtraction identifies regions of motion in the frame.
2. **Object classification** – YOLO models classify the detected motion regions as relevant object classes.
3. **Temporal tracking** – a lightweight tracker stabilizes detections across frames and filters one-frame false positives.
4. **Event-triggered recording** – confirmed objects trigger MP4 recording with pre-event and post-event context.
5. **Experiment logging** – an evaluator exports per-frame and aggregated metrics to CSV/JSON files for reproducible testing.

The implementation is intended as a research prototype for a bachelor thesis, not as a production-grade security product.

---

## Main features

- RTSP stream processing from a home IP camera
- Support for live stream and offline video-file evaluation
- MOG2 background subtraction for motion-region extraction
- YOLOv8n, YOLO11n and YOLO11s model comparison
- Motion-region and full-frame detection modes
- Temporal tracker with `TENTATIVE`, `CONFIRMED` and `LOST` states
- Event-triggered video recording with a circular pre-buffer
- CSV/JSON metric logging for experimental evaluation
- Reolink P340 built-in AI collector for comparative testing
- Reproducibility data for all thesis experiments

---

## Detection pipeline

```text
RTSP / video file
        |
        v
Frame preprocessing
        |
        v
MOG2 background subtraction
        |
        v
Contour filtering and motion-region extraction
        |
        v
YOLO object classification
        |
        v
Temporal tracking
        |
        v
Confirmed event?
        |
        +---- no  -> continue processing
        |
        +---- yes -> event recording + metrics export
```

Only `CONFIRMED` tracks are allowed to trigger security events and recordings. Tracks that are not matched for a configured number of frames transition to `LOST` and are then removed from the internal tracker registry.

---

## Hardware and tested environment

The reference deployment was tested on:

- Raspberry Pi 5, 8 GB RAM
- Raspberry Pi OS Bookworm 64-bit
- Reolink P340 IP camera
  - sub-stream: 896x512, H.264
  - main-stream: 4512x2512, H.265/HEVC
- Python 3.x
- OpenCV
- Ultralytics YOLO
- Gigabit Ethernet connection between the Raspberry Pi and the camera

Optional future extension:

- Raspberry Pi AI HAT+ with the Hailo-8 accelerator, not integrated in this prototype

The general detection pipeline does not require a specific camera model. Any RTSP-capable camera can be used if the RTSP URL and stream parameters are configured correctly. The `reolink_collector.py` module is Reolink-specific and is used only for the comparative experiment with the Reolink P340 built-in AI.

### Note on H.265/HEVC decoding

Raspberry Pi 5 includes a hardware HEVC/H.265 decoder. However, this prototype uses a simple OpenCV `VideoCapture` pipeline. In the implemented pipeline, H.265 hardware decoding was not practically used. Using hardware-accelerated decoding would require a dedicated video backend or pipeline, for example through GStreamer, V4L2, FFmpeg configuration, or a custom OpenCV build.

For this thesis prototype, the OpenCV-based pipeline was chosen because it is simple, reproducible and easy to integrate with frame-level MOG2 and YOLO processing. The trade-off is that high-resolution main-stream processing is CPU-heavy and is treated as a reference/offline configuration rather than a stable real-time deployment mode.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Muth008/motion-detection.git
cd motion-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a local environment file:

```bash
cp .env.example .env
```

Then edit `.env` and configure the camera connection:

```env
CAMERA_HOST=192.168.x.x
CAMERA_USERNAME=your_username
CAMERA_PASSWORD=your_password
```

The first run downloads the YOLO model weights automatically. The default model is configured in `config.py`.

---

## Usage

### Live operation

Run the main pipeline:

```bash
python main.py
```

The system connects to the configured RTSP stream and starts the detection pipeline.

Recorded events are saved to `./recordings/` as:

- MP4 video clips
- JPG thumbnails
- JSON metadata

For headless operation on Raspberry Pi, keep the debug windows disabled in `config.py`:

```python
SHOW_MAIN_WINDOW = False
SHOW_DEBUG_WINDOWS = False
```

### Experimental mode

For reproducible experiments, the system can replay a recorded video file instead of a live stream.

Record a test video:

```bash
python record_video.py --duration 300
```

Annotate ground-truth events:

```bash
python annotate_video.py test_video_<timestamp>_sub.mp4
```

Set `INPUT_SOURCE` in `config.py` to the recorded video file and run:

```bash
python main.py
```

Evaluate the session against the ground truth:

```bash
python evaluate_experiment.py ground_truth_<video>.csv --results eval_results/
```

### Comparison with Reolink built-in AI

The Reolink comparison is specific to the Reolink P340 camera and its built-in AI events.

Run both collectors in parallel:

```bash
# Terminal 1
python main.py
```

```bash
# Terminal 2
# Start once main.py reports that the sync file is ready.
python reolink_collector.py
```

Both processes share `session_sync.json`, which allows their outputs to be aligned in time.

The Reolink collector is used only for the comparative experiment described in the thesis. It is not required for normal operation with a generic RTSP camera.

---

## Configuration

All major parameters are centralized in `config.py`.

Main configuration groups:

- **Camera**
  - RTSP stream selection
  - sub-stream vs main-stream
  - credentials loaded from `.env`
- **MOG2 background subtraction**
  - history
  - variance threshold
  - shadow detection
- **Morphological filtering**
  - opening and closing kernel size
  - number of iterations
- **Contour filtering**
  - minimum/maximum area
  - width/height limits
  - aspect ratio
- **YOLO inference**
  - model file
  - confidence threshold
  - IoU threshold
  - detection mode: `motion_regions` or `full_frame`
- **Temporal tracker**
  - minimum hits before confirmation
  - maximum age before removal
  - IoU and distance thresholds
- **Recorder**
  - pre-buffer duration
  - post-buffer duration
  - codec
  - output directory
- **Evaluator**
  - session label
  - metrics output directory

The thesis experiments use several predefined parameter combinations. The exact settings are documented in the comments in `config.py`.

---

## Repository structure

```text
motion-detection/
├── main.py                  Main orchestration loop
├── config.py                Centralized configuration (.env-aware)
├── camera.py                RTSP / file video reader
├── motion_detector.py       MOG2 background subtraction
├── object_detector.py       YOLO wrapper based on Ultralytics
├── tracker.py               Lightweight temporal tracker
├── recorder.py              Event-triggered recording with circular buffer
├── evaluator.py             Per-frame metrics and CSV/JSON export
├── reolink_collector.py     Reolink HTTPS AI event collector
├── record_video.py          Standalone test-video recording utility
├── annotate_video.py        Ground-truth annotation tool
├── evaluate_experiment.py   TP/FP/FN and precision/recall/F1 evaluation
├── evaluator_data/          Raw data from all thesis experiments
│   ├── README.md            Description of experiment data files
│   ├── summary.csv          Aggregated results, one row per session
│   ├── sessions/            Per-frame CSV and JSON files
│   ├── ground_truth/        Manual annotations of test videos
│   └── results/             Evaluation outputs
├── requirements.txt         Python dependencies
├── .env.example             Template for camera credentials
├── .gitignore
└── LICENSE                  MIT license
```

---

## Experimental data and reproducibility

The repository contains the source code and the experimental data used in the thesis.

The `evaluator_data/` directory contains:

- `summary.csv` – aggregated metrics, one row per experimental session
- `sessions/` – per-frame CSV and JSON files for individual runs
- `ground_truth/` – manually annotated ground-truth events
- `results/` – TP/FP/FN and precision/recall/F1 evaluation outputs

The CSV files are not included as a physical appendix in the thesis document. Instead, the full experimental dataset is provided in this repository together with the implementation, so the results can be inspected and partially reproduced from the same source package.

To reproduce an experimental session:

1. Record or obtain a test video at the appropriate resolution.
2. Annotate ground-truth events with `annotate_video.py`.
3. Configure `config.py` according to the target experiment.
4. Run `python main.py` with the test video as `INPUT_SOURCE`.
5. Run `evaluate_experiment.py` against the corresponding ground-truth file.
6. Compare the generated CSV/JSON outputs with the files in `evaluator_data/`.

For thesis submission, the repository should ideally be referenced by a stable Git tag or commit hash, for example:

```bash
git tag bp-submission-v1
git push origin bp-submission-v1
```

---

## Experimental results summary

The thesis evaluates seven experimental groups based on a 14-event ground-truth dataset and additional false-alarm stress scenarios.

| Experiment | Compared setting | Key finding |
|---|---|---|
| Exp 1 | YOLO model: YOLO11n / YOLO11s / YOLOv8n | The tested models achieved the same recall on the selected dataset; the main bottleneck was upstream of YOLO. |
| Exp 2 | YOLO confidence threshold | F1 remained stable across the tested confidence range. |
| Exp 3 | Detection mode and image size | `motion_regions` matched full-frame detection with substantially lower YOLO latency. |
| Exp 4 | MOG2 sensitivity | Higher sensitivity recovered distant objects but increased false positives. |
| Exp 5 | Best sub-stream configuration | The sub-stream configuration was suitable for real-time prototyping but limited by input resolution. |
| Exp 6 | Main-stream reference evaluation | The high-resolution main-stream confirmed the potential of the architecture, but it was evaluated as a reference/offline configuration. |
| Exp 7 | False-alarm scenarios vs Reolink built-in AI | In the selected false-alarm stress scenarios, the proposed system produced no false alarms, while the Reolink built-in AI triggered false alarms in all person-free scenarios. |

The main finding is that, on the low-resolution sub-stream, the detection ceiling was primarily caused by input resolution and MOG2 contour generation, not by the YOLO classifier itself. The main-stream results indicate that higher input resolution can substantially improve detection quality, but practical real-time main-stream processing would require further optimization of the video pipeline and/or hardware acceleration.

The comparison with the Reolink P340 built-in AI should be interpreted as a targeted false-alarm stress test, not as a universal benchmark of all Reolink camera functionality or all possible home surveillance scenarios.

---

## Security and credentials

Camera credentials are loaded from the local `.env` file and should never be committed to the repository.

The included `.env.example` file contains only placeholder values. The Reolink collector communicates with the camera in the local network and is intended for controlled experimental use only.

Before publishing or submitting modified versions of this repository, verify that no private credentials, IP addresses or personal video recordings have been committed.

---

## Thesis submission version

Recommended citation of the code version:

- Repository: `https://github.com/Muth008/motion-detection`
- Submission tag: `bp-submission-v1`
- If no tag is available, cite the exact commit hash used for the thesis submission.

---

## Citation

If you use this code or its experimental results, please cite the thesis:

> Šindelář, J. (2026). *Detekce a klasifikace pohybu na domácích IP kamerách s využitím edge AI a filtrací falešných poplachů*. Bachelor thesis, Unicorn University, Prague.

---

## License

MIT License. See `LICENSE` for details.
