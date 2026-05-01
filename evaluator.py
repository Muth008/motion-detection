#!/usr/bin/env python3
"""
Performance metrics collector and exporter for the detection system.

Three areas are measured:
    1. Pipeline throughput - how many detections pass through each stage
       (this is the basis for false-alarm-reduction calculations in the thesis).
    2. Latency - time spent in each stage (MOG2, YOLO, tracker).
    3. System resources - FPS, CPU, RAM.

False Alarm Reduction (FAR) calculation:
    FAR YOLO    = % of motion frames where YOLO did not find any relevant object
                = frames(motion>0 AND yolo_ran AND yolo==0) / frames(motion>0 AND yolo_ran)
    FAR Tracker = % of YOLO-detection frames where the tracker did not
                  newly confirm any object
                = frames(yolo>0 AND new_confirmations==0) / frames(yolo>0)

    This formulation is correct because the tracker keeps tracks alive after
    the object disappears (TRACKER_MAX_AGE), so confirmed_tracks cannot be
    compared directly to yolo_detections.

Session synchronization:
    The evaluator writes ``session_sync.json`` to EVAL_OUTPUT_DIR. The
    ReolinkCollector reads this file and stores the same start_time in its
    own JSON output. This makes it explicit that both measurements happened
    in the same time window.

Output files (in EVAL_OUTPUT_DIR):
    <ts>_<session>.csv              - per-frame records
    <ts>_<session>_summary.json     - aggregated statistics for the thesis
    session_sync.json               - shared sync file (overwritten on start)
"""

import csv
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

import config


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class FrameRecord:
    """Metrics record for a single processed frame."""
    timestamp: float
    frame_index: int

    # Pipeline throughput
    motion_regions: int       # MOG2 output (number of motion regions)
    yolo_detections: int      # YOLO output (number of raw detections)
    confirmed_tracks: int     # currently alive confirmed tracks
    new_confirmations: int    # tracks confirmed for the first time IN THIS frame
                              # -> the key signal for the tracker FAR formula

    # Per-stage latency [ms]
    time_mog2_ms: float
    time_yolo_ms: float
    time_tracker_ms: float

    # System metrics
    fps: float
    cpu_percent: float
    ram_mb: float

    # Was YOLO actually run on this frame?
    yolo_ran: bool


@dataclass
class SessionSummary:
    """Aggregated statistics for an entire measurement session."""
    session_label: str
    model_name: str
    camera_name: str
    start_time: str
    sync_start_time: str = ""   # shared start (Evaluator's start_time)
    end_time: str = ""
    duration_s: float = 0.0
    total_frames: int = 0

    # -- Absolute frame counts (basis for FAR calculations, transparent) ------
    frames_with_motion: int = 0       # frames where MOG2 detected motion
    frames_yolo_ran: int = 0          # frames where YOLO was executed
    frames_yolo_detected: int = 0     # frames where YOLO returned >=1 detection
    frames_tracker_confirmed: int = 0 # frames where the tracker newly confirmed an object

    # -- Pipeline averages ----------------------------------------------------
    avg_motion_regions: float = 0.0
    avg_yolo_detections: float = 0.0
    avg_confirmed_tracks: float = 0.0

    # -- False Alarm Reduction ------------------------------------------------
    false_alarm_reduction_yolo_pct: float = 0.0
    false_alarm_reduction_tracker_pct: float = 0.0

    # -- Latency [ms] ---------------------------------------------------------
    avg_mog2_ms: float = 0.0
    avg_yolo_ms: float = 0.0
    avg_tracker_ms: float = 0.0
    p95_yolo_ms: float = 0.0
    p99_yolo_ms: float = 0.0

    # -- System metrics -------------------------------------------------------
    avg_fps: float = 0.0
    avg_cpu_pct: float = 0.0
    avg_ram_mb: float = 0.0
    yolo_call_rate_pct: float = 0.0

    # -- Events ---------------------------------------------------------------
    total_confirmed_events: int = 0
    total_recordings: int = 0


# =============================================================================
# Session sync file
# =============================================================================

class SessionSync:
    """
    Shared sync file used by the evaluator and the ReolinkCollector to align
    on a common start time.

    The evaluator writes the file in ``start()`` and the collector reads it.

    File format (session_sync.json):
        {
            "session_label":   "yolo11n_outdoor_day",
            "start_time":      "2025-03-07T11:34:57.834989",
            "start_timestamp": 1741346097.834989
        }
    """

    SYNC_FILENAME = "session_sync.json"

    @classmethod
    def write(cls, session_label: str, start_timestamp: float, output_dir: Path) -> Path:
        """Write the sync file. Called by the evaluator on start()."""
        data = {
            "session_label": session_label,
            "start_time": datetime.fromtimestamp(start_timestamp).isoformat(),
            "start_timestamp": start_timestamp,
        }
        path = output_dir / cls.SYNC_FILENAME
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    @classmethod
    def read(cls, output_dir: Path) -> Optional[dict]:
        """Read the sync file. Called by the collector on start()."""
        path = output_dir / cls.SYNC_FILENAME
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None


# =============================================================================
# Main class
# =============================================================================

class Evaluator:
    """
    Metrics collector for the experimental section of the thesis.

    The overhead of ``record_frame()`` is minimal - the critical path only
    appends to a list. CSV writing happens in a background thread.
    """

    def __init__(self, session_label: str = ""):
        self.session_label = session_label or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(config.EVAL_OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.output_dir / f"{ts}_{self.session_label}.csv"
        self.json_path = self.output_dir / f"{ts}_{self.session_label}_summary.json"

        self._buffer: list = []
        self._buffer_lock = threading.Lock()
        self._all_records: list = []

        self._frame_index = 0
        self._start_time = 0.0
        self._running = False
        self._writer_thread: Optional[threading.Thread] = None
        self._csv_writer = None
        self._csv_file = None

        self.total_confirmed_events = 0
        self.total_recordings = 0

        self._process = psutil.Process(os.getpid()) if PSUTIL_AVAILABLE else None

    def start(self):
        """Open the CSV file, write the sync file and start the writer thread."""
        self._start_time = time.time()
        self._running = True

        # Sync file - the collector reads this and uses it as the reference start
        sync_path = SessionSync.write(
            self.session_label, self._start_time, self.output_dir
        )
        print(f"   [SYNC] file: {sync_path.name}  <- start reolink_collector.py now")

        # CSV
        self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        fieldnames = list(FrameRecord.__dataclass_fields__.keys())
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
        self._csv_writer.writeheader()

        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

        print(f"   [EVAL] started | CSV: {self.csv_path.name}")
        if not PSUTIL_AVAILABLE:
            print("   [WARN] psutil missing - CPU/RAM will not be measured (pip install psutil)")

    def stop(self) -> SessionSummary:
        """Stop collection, flush remaining records and write the summary JSON."""
        self._running = False
        if self._writer_thread:
            self._writer_thread.join(timeout=3.0)
        self._flush_buffer()
        if self._csv_file:
            self._csv_file.close()

        summary = self._compute_summary()
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(summary), f, indent=2, ensure_ascii=False)

        n = max(summary.total_frames, 1)
        print("\n[EVAL] Session terminated")
        print(f"   Total frames:          {summary.total_frames}")
        print(f"   Average FPS:           {summary.avg_fps:.1f}")
        print(f"   Frames with motion:    {summary.frames_with_motion} "
              f"({summary.frames_with_motion / n * 100:.1f} %)")
        print(f"   FAR reduction (YOLO):  {summary.false_alarm_reduction_yolo_pct:.1f} %")
        print(f"   FAR reduction (Track): {summary.false_alarm_reduction_tracker_pct:.1f} %")
        print(f"   Avg YOLO latency:      {summary.avg_yolo_ms:.1f} ms "
              f"(p95: {summary.p95_yolo_ms:.1f} ms)")
        print(f"   CSV:  {self.csv_path}")
        print(f"   JSON: {self.json_path}")

        return summary

    # -------------------------------------------------------------------------
    # Recording API (called from the main loop)
    # -------------------------------------------------------------------------

    def record_frame(
        self,
        motion_regions: int,
        yolo_detections: int,
        confirmed_tracks: int,
        new_confirmations: int,
        stage_times: dict,
        fps: float,
        yolo_ran: bool = True,
    ):
        """
        Record metrics for a single frame.

        Args:
            motion_regions:    number of MOG2 motion regions
            yolo_detections:   number of raw YOLO detections
            confirmed_tracks:  currently alive confirmed tracks
            new_confirmations: tracks confirmed for the first time in this frame
                               (track.hits == TRACKER_MIN_HITS)
            stage_times:       {"mog2": s, "yolo": s, "tracker": s}
            fps:               current FPS
            yolo_ran:          True if YOLO was actually executed on this frame
        """
        cpu = self._process.cpu_percent() if self._process else 0.0
        ram = self._process.memory_info().rss / 1024 / 1024 if self._process else 0.0

        record = FrameRecord(
            timestamp=time.time(),
            frame_index=self._frame_index,
            motion_regions=motion_regions,
            yolo_detections=yolo_detections,
            confirmed_tracks=confirmed_tracks,
            new_confirmations=new_confirmations,
            time_mog2_ms=stage_times.get("mog2", 0.0) * 1000,
            time_yolo_ms=stage_times.get("yolo", 0.0) * 1000,
            time_tracker_ms=stage_times.get("tracker", 0.0) * 1000,
            fps=fps,
            cpu_percent=cpu,
            ram_mb=ram,
            yolo_ran=yolo_ran,
        )
        self._frame_index += 1
        with self._buffer_lock:
            self._buffer.append(record)
            self._all_records.append(record)

    def increment_event(self):
        """Increment the confirmed-events counter."""
        self.total_confirmed_events += 1

    def increment_recording(self):
        """Increment the saved-recordings counter."""
        self.total_recordings += 1

    # -------------------------------------------------------------------------
    # Internal methods
    # -------------------------------------------------------------------------

    def _writer_loop(self):
        while self._running:
            self._flush_buffer()
            time.sleep(0.1)

    def _flush_buffer(self):
        with self._buffer_lock:
            to_write = list(self._buffer)
            self._buffer.clear()
        if to_write and self._csv_writer:
            for record in to_write:
                self._csv_writer.writerow(asdict(record))
            self._csv_file.flush()

    def _compute_summary(self) -> SessionSummary:
        records = self._all_records
        n = len(records)

        sync = SessionSync.read(self.output_dir)
        sync_start = sync.get("start_time", "") if sync else ""

        summary = SessionSummary(
            session_label=self.session_label,
            model_name=config.YOLO_MODEL,
            camera_name=config.CAMERA_NAME,
            start_time=datetime.fromtimestamp(self._start_time).isoformat(),
            sync_start_time=sync_start,
            end_time=datetime.now().isoformat(),
            duration_s=round(time.time() - self._start_time, 1),
            total_frames=n,
            total_confirmed_events=self.total_confirmed_events,
            total_recordings=self.total_recordings,
        )

        if n == 0:
            return summary

        def avg(lst):
            return sum(lst) / len(lst) if lst else 0.0

        def percentile(lst, p):
            if not lst:
                return 0.0
            s = sorted(lst)
            return s[min(int(len(s) * p / 100), len(s) - 1)]

        # -- Frame groups ------------------------------------------------------
        frames_with_motion = [r for r in records if r.motion_regions > 0]
        frames_yolo_ran = [r for r in records if r.yolo_ran]
        frames_yolo_detected = [r for r in records if r.yolo_ran and r.yolo_detections > 0]
        frames_tracker_conf = [r for r in records if r.new_confirmations > 0]

        summary.frames_with_motion = len(frames_with_motion)
        summary.frames_yolo_ran = len(frames_yolo_ran)
        summary.frames_yolo_detected = len(frames_yolo_detected)
        summary.frames_tracker_confirmed = len(frames_tracker_conf)

        # -- Averages ----------------------------------------------------------
        summary.avg_motion_regions = round(avg([r.motion_regions for r in records]), 2)
        summary.avg_yolo_detections = round(avg([r.yolo_detections for r in records]), 2)
        summary.avg_confirmed_tracks = round(avg([r.confirmed_tracks for r in records]), 2)

        # -- False Alarm Reduction --------------------------------------------
        # FAR YOLO:
        #   Of all frames where MOG2 saw motion AND YOLO ran, in what % did
        #   YOLO find no relevant object?
        yolo_ran_on_motion = [r for r in frames_with_motion if r.yolo_ran]
        if yolo_ran_on_motion:
            yolo_found_nothing = sum(1 for r in yolo_ran_on_motion if r.yolo_detections == 0)
            summary.false_alarm_reduction_yolo_pct = round(
                yolo_found_nothing / len(yolo_ran_on_motion) * 100, 1)

        # FAR Tracker:
        #   Of all frames where YOLO detected something, in what % did the
        #   tracker fail to newly confirm any object? (= unstable / flickering)
        if frames_yolo_detected:
            tracker_no_new = sum(1 for r in frames_yolo_detected if r.new_confirmations == 0)
            summary.false_alarm_reduction_tracker_pct = round(
                tracker_no_new / len(frames_yolo_detected) * 100, 1)

        # -- Latency -----------------------------------------------------------
        yolo_times = [r.time_yolo_ms for r in frames_yolo_ran if r.time_yolo_ms > 0]
        summary.avg_mog2_ms = round(avg([r.time_mog2_ms for r in records]), 2)
        summary.avg_yolo_ms = round(avg(yolo_times), 2)
        summary.avg_tracker_ms = round(avg([r.time_tracker_ms for r in records]), 2)
        summary.p95_yolo_ms = round(percentile(yolo_times, 95), 2)
        summary.p99_yolo_ms = round(percentile(yolo_times, 99), 2)

        # -- System metrics ----------------------------------------------------
        summary.avg_fps = round(avg([r.fps for r in records if r.fps > 0]), 2)
        summary.avg_cpu_pct = round(avg([r.cpu_percent for r in records]), 2)
        summary.avg_ram_mb = round(avg([r.ram_mb for r in records]), 2)
        summary.yolo_call_rate_pct = round(len(frames_yolo_ran) / n * 100, 1)

        return summary

    def get_live_stats(self) -> dict:
        """Live stats (last 100 frames) for console output and overlay."""
        with self._buffer_lock:
            recent = self._all_records[-100:]
            total = len(self._all_records)
        if not recent:
            return {}

        n = len(recent)
        yolo_ran_on_motion = [r for r in recent if r.motion_regions > 0 and r.yolo_ran]
        yolo_times = [r.time_yolo_ms for r in recent if r.yolo_ran and r.time_yolo_ms > 0]
        cpu_vals = [r.cpu_percent for r in recent if r.cpu_percent > 0]
        ram_vals = [r.ram_mb for r in recent if r.ram_mb > 0]

        far = 0.0
        if yolo_ran_on_motion:
            far = (sum(1 for r in yolo_ran_on_motion if r.yolo_detections == 0)
                   / len(yolo_ran_on_motion) * 100)

        def pct(lst, p):
            if not lst:
                return 0.0
            s = sorted(lst)
            return s[min(int(len(s) * p / 100), len(s) - 1)]

        return {
            "frames": total,
            "frames_with_motion": sum(1 for r in self._all_records if r.motion_regions > 0),
            "avg_mog2_ms": round(sum(r.time_mog2_ms for r in recent) / n, 1),
            "avg_yolo_ms": round(sum(yolo_times) / len(yolo_times), 1) if yolo_times else 0,
            "p95_yolo_ms": round(pct(yolo_times, 95), 1),
            "avg_fps": round(sum(r.fps for r in recent if r.fps > 0) / n, 1),
            "far_yolo": round(far, 1),
            "avg_cpu_pct": round(sum(cpu_vals) / len(cpu_vals), 1) if cpu_vals else 0,
            "avg_ram_mb": round(sum(ram_vals) / len(ram_vals), 1) if ram_vals else 0,
        }
