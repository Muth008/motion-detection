#!/usr/bin/env python3
"""
Configuration of the motion detection system - profile for Raspberry Pi 5.

Sensitive credentials (camera password, IP address) are loaded from the
``.env`` file using ``python-dotenv``. Copy ``.env.example`` to ``.env`` and
fill in your actual values before running the application.

ACTIVE EXPERIMENT - switched manually before each session:
  Exp 1 - Model effect:        change YOLO_MODEL + EVAL_SESSION_LABEL
  Exp 2 - Confidence effect:   change YOLO_CONFIDENCE_THRESHOLD + EVAL_SESSION_LABEL
  Exp 3 - Detection mode:      change YOLO_DETECTION_MODE (+ YOLO_IMGSZ) + EVAL_SESSION_LABEL
  Exp 4 - Stream effect:       change RTSP_STREAM + PROCESSING_WIDTH + FRAME_SKIP + EVAL_SESSION_LABEL

Workflow for each session:
  Terminal 1: python main.py
  Terminal 2: python reolink_collector.py
  -> After session, copy eval_results/ to backup and change EVAL_SESSION_LABEL.
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv is optional; if missing, environment variables must be set manually
    pass


# =============================================================================
# CAMERA - credentials and connection details loaded from .env
# =============================================================================
CAMERA_NAME = os.environ.get("CAMERA_NAME", "porch")

CAMERA_HOST = os.environ.get("CAMERA_HOST", "192.168.1.100")
CAMERA_USERNAME = os.environ.get("CAMERA_USERNAME", "admin")
CAMERA_PASSWORD = os.environ.get("CAMERA_PASSWORD", "")
CAMERA_RTSP_PORT = int(os.environ.get("CAMERA_RTSP_PORT", "554"))

# Stream selection: "sub" (low-resolution H.264) or "main" (high-resolution H.265)
RTSP_STREAM = os.environ.get("RTSP_STREAM", "sub")

# Build the RTSP URL from the components above
_stream_path = "h264Preview_01_sub" if RTSP_STREAM == "sub" else "h264Preview_01_main"
RTSP_URL = (
    f"rtsp://{CAMERA_USERNAME}:{CAMERA_PASSWORD}"
    f"@{CAMERA_HOST}:{CAMERA_RTSP_PORT}/{_stream_path}"
)


# =============================================================================
# EXPERIMENTAL MODE - input from a video file
#
# Plays a local video file instead of the live RTSP stream.
# Advantage: identical conditions for all experiments (Exp 1-3).
#
# Usage:
#   INPUT_SOURCE = "./test_video.mp4"   -> plays the file
#   INPUT_SOURCE = None                 -> uses RTSP_URL (live stream)
#
# PLAYBACK_SPEED: 1.0 = real time, 2.0 = 2x faster
#   At higher playback speeds MOG2 may behave differently - for experiments
#   use 1.0 or at most 1.5 (otherwise the object "jumps" too far between frames)
# =============================================================================
INPUT_SOURCE = None            # None = live RTSP | "./test_video.mp4" = file
PLAYBACK_SPEED = 1.0           # file playback speed multiplier


# =============================================================================
# PREPROCESSING
# =============================================================================
# Sub-stream: 500, Main-stream: 1200
PROCESSING_WIDTH = 500
BLUR_KERNEL_SIZE = 5

# FRAME_SKIP: process every N-th frame (YOLO+MOG2).
# Sub-stream RPi: 2  |  Main-stream RPi: 4  |  Mac: 3
FRAME_SKIP = 2


# =============================================================================
# MOG2 BACKGROUND SUBTRACTION
# =============================================================================
MOG2_HISTORY = 500
MOG2_VAR_THRESHOLD = 25        # 25 = more sensitive (distant persons); 50 = fewer FPs
MOG2_DETECT_SHADOWS = True
MOG2_SHADOW_VALUE = 127
MOG2_LEARNING_RATE = -1


# =============================================================================
# MORPHOLOGICAL OPERATIONS
# =============================================================================
MORPH_KERNEL_SIZE = 5
MORPH_OPEN_ITERATIONS = 2
MORPH_CLOSE_ITERATIONS = 2


# =============================================================================
# CONTOUR FILTERING
# =============================================================================
MIN_CONTOUR_AREA = 600
MAX_CONTOUR_AREA = 50000
MIN_WIDTH = 20
MIN_HEIGHT = 20
MIN_ASPECT_RATIO = 0.2
MAX_ASPECT_RATIO = 5.0


# =============================================================================
# YOLO OBJECT DETECTION
#
# -- Experiment 1: Model effect (fixed: motion_regions, conf 0.4) -------------
#   Session "yolo11n_sub_exp1":  YOLO_MODEL = "yolo11n.pt"
#   Session "yolo11s_sub_exp1":  YOLO_MODEL = "yolo11s.pt"
#   Session "yolov8n_sub_exp1":  YOLO_MODEL = "yolov8n.pt"
#
# -- Experiment 2: Confidence effect (fixed: yolo11s, motion_regions) ---------
#   Session "conf025_sub_exp2":  YOLO_CONFIDENCE_THRESHOLD = 0.25
#   Session "conf040_sub_exp2":  YOLO_CONFIDENCE_THRESHOLD = 0.40
#   Session "conf055_sub_exp2":  YOLO_CONFIDENCE_THRESHOLD = 0.55
#
# -- Experiment 3: Detection mode effect (fixed: yolo11s, conf 0.4) -----------
#   Session "regions_sub_exp3":  YOLO_DETECTION_MODE = "motion_regions", YOLO_IMGSZ = 640
#   Session "full640_sub_exp3":  YOLO_DETECTION_MODE = "full_frame",     YOLO_IMGSZ = 640
#   Session "full1280_sub_exp3": YOLO_DETECTION_MODE = "full_frame",     YOLO_IMGSZ = 1280
#
# -- Experiment 4: Stream effect (fixed: yolo11s, conf 0.4, motion_regions) ---
#   Session "sub_stream_exp4":  RTSP sub, PROCESSING_WIDTH=500,  FRAME_SKIP=2
#   Session "main_stream_exp4": RTSP main, PROCESSING_WIDTH=1200, FRAME_SKIP=4
# =============================================================================
YOLO_MODEL = "yolo11n.pt"
YOLO_CONFIDENCE_THRESHOLD = 0.40
YOLO_IOU_THRESHOLD = 0.45
YOLO_FILTER_CLASSES = True
# COCO class indices: person, bicycle, car, motorcycle, bus, truck, cat, dog
YOLO_RELEVANT_CLASSES = [0, 1, 2, 3, 5, 7, 15, 16]
YOLO_REGION_PADDING = 80
YOLO_DETECTION_MODE = "motion_regions"   # "motion_regions" | "full_frame"
YOLO_ONLY_ON_MOTION = True
YOLO_IMGSZ = 640                         # 640 (default) | 1280 (better recall, ~2x slower)


# =============================================================================
# TEMPORAL TRACKING
# =============================================================================
TRACKER_MIN_HITS = 2
TRACKER_MAX_AGE = 15
TRACKER_IOU_THRESHOLD = 0.05   # low value tolerates larger movement when FRAME_SKIP > 1
TRACKER_MAX_HISTORY = 30
TRACKER_TENTATIVE_MAX_AGE = 6
TRACKER_LOST_MAX_AGE = 30
TRACKER_SHOW_TENTATIVE = False


# =============================================================================
# EVENT RECORDING
#
# RECORDING_BUFFER_WIDTH/HEIGHT: buffer stores HD frames instead of full stream
# resolution -> less RAM usage. The detection pipeline (YOLO/MOG2) still works
# with the full-resolution frame.
# =============================================================================
RECORDING_OUTPUT_DIR = "./recordings"
RECORDING_PRE_BUFFER = 2.0
RECORDING_POST_BUFFER = 2.0
RECORDING_THUMBNAIL_DELAY = 2.0
RECORDING_CODEC = "mp4v"
RECORDING_MAX_DURATION = 300
RECORDING_ENABLED = True
RECORDING_BUFFER_WIDTH = 896    # Sub-stream native width; main-stream: 1280
RECORDING_BUFFER_HEIGHT = 512   # Sub-stream native height; main-stream: 720


# =============================================================================
# REOLINK COLLECTOR (Section 4.5 of the thesis)
# Always run in parallel:
#   Terminal 1: python main.py
#   Terminal 2: python reolink_collector.py
# =============================================================================
REOLINK_HOST = os.environ.get("REOLINK_HOST", CAMERA_HOST)
REOLINK_PORT = int(os.environ.get("REOLINK_PORT", "443"))
REOLINK_USERNAME = os.environ.get("REOLINK_USERNAME", CAMERA_USERNAME)
REOLINK_PASSWORD = os.environ.get("REOLINK_PASSWORD", CAMERA_PASSWORD)
REOLINK_CHANNEL = int(os.environ.get("REOLINK_CHANNEL", "0"))
REOLINK_POLL_INTERVAL = 1.0


# =============================================================================
# EVALUATOR
#
# EVAL_SESSION_LABEL - change before each session according to the scheme
# in the YOLO section comments above.
# Output files: eval_results/<ts>_<label>.csv + <ts>_<label>_summary.json
# =============================================================================
EVAL_ENABLED = True
EVAL_OUTPUT_DIR = "./eval_results"
EVAL_SESSION_LABEL = "yolo11n_sub_exp1"   # <- CHANGE before each session


# =============================================================================
# DISPLAY
# RPi without monitor: both False (default)
# Mac with monitor:    both True
# =============================================================================
SHOW_MAIN_WINDOW = False    # True for Mac, False for headless RPi
SHOW_DEBUG_WINDOWS = False  # True for Mac, False for headless RPi
WINDOW_WAIT_MS = 1
STATS_PRINT_INTERVAL = 30   # print live stats to console every N seconds

COLORS = {
    "motion":    (0, 255, 0),
    "person":    (0, 0, 255),
    "vehicle":   (255, 165, 0),
    "animal":    (255, 0, 255),
    "other":     (255, 255, 0),
    "recording": (0, 0, 255),
}
