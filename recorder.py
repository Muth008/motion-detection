#!/usr/bin/env python3
"""
Event-triggered video recording.

Features:
    - A circular buffer keeps the last N seconds of frames in memory
      (pre-buffer).
    - When a CONFIRMED event is detected the recorder starts writing.
    - After the event ends the recorder keeps writing for another M seconds
      (post-buffer).
    - Final outputs: MP4 video, JPG thumbnail and JSON metadata file.

How it works:
    1. The circular buffer continuously stores frames (FIFO).
    2. On a CONFIRMED detection -> start recording.
    3. The saved clip = pre-buffer + active recording + post-buffer.
    4. The result is a single video that captures context before, during
       and after the event.

Example:
    Pre-buffer: 5s | Event: 10s | Post-buffer: 5s
    Saved clip: 20s
"""

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config


class RecordingState(Enum):
    IDLE = "idle"               # Waiting for an event
    RECORDING = "recording"     # Active recording
    POST_BUFFER = "post_buffer" # Recording the post-event tail


@dataclass
class EventMetadata:
    """Metadata describing one recorded event."""
    event_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: float = 0.0
    trigger_class: str = ""
    trigger_track_id: int = 0
    trigger_confidence: float = 0.0
    max_objects: int = 0
    classes_detected: List[str] = field(default_factory=list)
    video_path: str = ""
    thumbnail_path: str = ""

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "trigger_class": self.trigger_class,
            "trigger_track_id": self.trigger_track_id,
            "trigger_confidence": self.trigger_confidence,
            "max_objects": self.max_objects,
            "classes_detected": self.classes_detected,
            "video_path": self.video_path,
            "thumbnail_path": self.thumbnail_path,
        }


class CircularBuffer:
    """
    FIFO frame buffer with a fixed capacity.

    Holds the most recent N frames in memory. When the buffer is full the
    oldest frame is overwritten.
    """

    def __init__(self, max_frames: int):
        self.buffer = deque(maxlen=max_frames)
        self.timestamps = deque(maxlen=max_frames)

    def add(self, frame: np.ndarray, timestamp: float = None):
        if timestamp is None:
            timestamp = time.time()
        # Copy the frame to avoid reference-related issues if the caller mutates it
        self.buffer.append(frame.copy())
        self.timestamps.append(timestamp)

    def get_all(self) -> List[Tuple[np.ndarray, float]]:
        """Return all (frame, timestamp) tuples currently in the buffer."""
        return list(zip(self.buffer, self.timestamps))

    def clear(self):
        self.buffer.clear()
        self.timestamps.clear()

    def __len__(self) -> int:
        return len(self.buffer)

    @property
    def is_full(self) -> bool:
        return len(self.buffer) == self.buffer.maxlen


class EventRecorder:
    """
    Event-triggered video recorder.

    Records a video clip when an event is detected, including pre- and
    post-event context.

    Usage:
        recorder = EventRecorder()

        while True:
            frame = camera.read()
            confirmed_tracks = tracker.update(detections)

            recorder.add_frame(frame)
            recorder.update(confirmed_tracks)
    """

    def __init__(self, output_dir: str = None, fps: float = 15.0):
        """
        Args:
            output_dir: Directory for video output (default from config).
            fps:        Expected camera frame rate (used for accurate
                        pre-buffer sizing and video file FPS).
        """
        self.output_dir = Path(output_dir or config.RECORDING_OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.estimated_fps = fps if fps > 0 else 15.0
        pre_buffer_frames = int(config.RECORDING_PRE_BUFFER * self.estimated_fps)

        self.pre_buffer = CircularBuffer(max_frames=pre_buffer_frames)
        self.recording_buffer: List[Tuple[np.ndarray, float]] = []

        self.state = RecordingState.IDLE
        self.current_event: Optional[EventMetadata] = None
        self.post_buffer_start: float = 0
        self.last_detection_time: float = 0
        self.trigger_timestamp: float = 0.0

        self.video_writer: Optional[cv2.VideoWriter] = None
        self.frame_size: Optional[Tuple[int, int]] = None

        # Snapshot of confirmed tracks captured around the thumbnail timestamp
        self.thumbnail_tracks: list = []
        self.total_events = 0
        self.total_duration = 0.0

        self.lock = threading.Lock()

        print(f"   Output dir: {self.output_dir}")
        print(f"   Pre-buffer:  {config.RECORDING_PRE_BUFFER}s (~{pre_buffer_frames} frames)")
        print(f"   Post-buffer: {config.RECORDING_POST_BUFFER}s")

    def add_frame(self, frame: np.ndarray):
        """
        Append a frame to the active buffer.

        IDLE state:                appended to the pre-buffer.
        RECORDING / POST_BUFFER:   appended to the active recording buffer.
        """
        current_time = time.time()

        if self.frame_size is None:
            self.frame_size = (frame.shape[1], frame.shape[0])

        with self.lock:
            if self.state == RecordingState.IDLE:
                self.pre_buffer.add(frame, current_time)
            elif self.state in (RecordingState.RECORDING, RecordingState.POST_BUFFER):
                self.recording_buffer.append((frame.copy(), current_time))

    def update(self, confirmed_tracks: list) -> Optional[str]:
        """
        Drive the recording state machine using the current set of
        confirmed tracks.

        Returns:
            Path to the saved video if a clip was just finalized, else None.
        """
        current_time = time.time()
        has_detection = len(confirmed_tracks) > 0

        with self.lock:
            # === IDLE -> RECORDING ===
            if self.state == RecordingState.IDLE:
                if has_detection:
                    self._start_recording(confirmed_tracks, current_time)

            # === RECORDING ===
            elif self.state == RecordingState.RECORDING:
                if has_detection:
                    # Save tracks while we are around the thumbnail moment
                    if current_time <= (self.trigger_timestamp
                                        + config.RECORDING_THUMBNAIL_DELAY + 0.5):
                        self.thumbnail_tracks = self._copy_tracks(confirmed_tracks)
                    self.last_detection_time = current_time
                    self._update_metadata(confirmed_tracks)
                else:
                    # No detection -> enter post-buffer
                    self.state = RecordingState.POST_BUFFER
                    self.post_buffer_start = current_time

            # === POST_BUFFER ===
            elif self.state == RecordingState.POST_BUFFER:
                if has_detection:
                    # Detection resumed -> back to RECORDING
                    self.state = RecordingState.RECORDING
                    self.last_detection_time = current_time
                    self._update_metadata(confirmed_tracks)
                elif current_time - self.post_buffer_start >= config.RECORDING_POST_BUFFER:
                    # Post-buffer expired -> save the video
                    return self._finish_recording(current_time)

        return None

    def _start_recording(self, tracks: list, current_time: float):
        """Begin a new recording."""
        self.state = RecordingState.RECORDING
        self.last_detection_time = current_time
        self.trigger_timestamp = current_time
        self.thumbnail_tracks = self._copy_tracks(tracks)

        event_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        trigger_track = tracks[0] if tracks else None

        self.current_event = EventMetadata(
            event_id=event_id,
            start_time=datetime.now(),
            trigger_class=trigger_track.class_name if trigger_track else "",
            trigger_track_id=trigger_track.track_id if trigger_track else 0,
            trigger_confidence=trigger_track.median_confidence if trigger_track else 0,
        )

        # Move the pre-buffer into the active recording buffer
        self.recording_buffer = list(self.pre_buffer.get_all())
        self.pre_buffer.clear()

        self._update_metadata(tracks)

        print(
            f"[RECORDING] Started: {event_id} "
            f"(trigger: {self.current_event.trigger_class})"
        )

    def _copy_tracks(self, tracks: list) -> list:
        """Snapshot only the data we need so the tracker can mutate freely."""
        copied = []
        for track in tracks:
            if track.is_confirmed():
                copied.append({
                    "bbox": track.current_bbox,
                    "class_name": track.class_name,
                    "confidence": track.median_confidence,
                })
        return copied

    def _update_metadata(self, tracks: list):
        """Update running metadata while the event is in progress."""
        if not self.current_event:
            return
        if len(tracks) > self.current_event.max_objects:
            self.current_event.max_objects = len(tracks)
        for track in tracks:
            if track.class_name not in self.current_event.classes_detected:
                self.current_event.classes_detected.append(track.class_name)

    def _finish_recording(self, current_time: float) -> Optional[str]:
        """Finalize and persist the current recording."""
        if not self.current_event or not self.recording_buffer:
            self._reset_state()
            return None

        self.current_event.end_time = datetime.now()

        # Compute the actual duration from the first/last frame timestamps
        if len(self.recording_buffer) >= 2:
            first_ts = self.recording_buffer[0][1]
            last_ts = self.recording_buffer[-1][1]
            self.current_event.duration = last_ts - first_ts

        video_filename = f"{self.current_event.event_id}.mp4"
        thumbnail_filename = f"{self.current_event.event_id}_thumb.jpg"
        metadata_filename = f"{self.current_event.event_id}.json"

        video_path = self.output_dir / video_filename
        thumbnail_path = self.output_dir / thumbnail_filename
        metadata_path = self.output_dir / metadata_filename

        self.current_event.video_path = str(video_path)
        self.current_event.thumbnail_path = str(thumbnail_path)

        # Use the camera-estimated FPS to avoid the YOLO bottleneck speeding up
        # the saved video (real processing rate may be much lower than incoming).
        actual_fps = self.estimated_fps

        try:
            self._save_video(video_path, actual_fps)
            self._save_thumbnail(thumbnail_path)
            self._save_metadata(metadata_path)

            print(
                f"[RECORDING] Saved: {video_filename} "
                f"({self.current_event.duration:.1f}s, "
                f"{len(self.recording_buffer)} frames)"
            )

            self.total_events += 1
            self.total_duration += self.current_event.duration
            saved_path = str(video_path)

        except Exception as e:
            print(f"[RECORDING] Error saving: {e}")
            saved_path = None

        self._reset_state()
        return saved_path

    def _save_video(self, path: Path, fps: float):
        """Write the recorded frames into an MP4 file."""
        if not self.recording_buffer or not self.frame_size:
            return

        fourcc = cv2.VideoWriter_fourcc(*config.RECORDING_CODEC)
        writer = cv2.VideoWriter(str(path), fourcc, fps, self.frame_size)

        if not writer.isOpened():
            raise RuntimeError(f"Cannot open video writer for {path}")

        for frame, _ in self.recording_buffer:
            writer.write(frame)

        writer.release()

    def _save_thumbnail(self, path: Path):
        """
        Save a thumbnail JPG taken roughly at
        ``trigger_timestamp + RECORDING_THUMBNAIL_DELAY``.
        """
        if not self.recording_buffer:
            return

        target_timestamp = self.trigger_timestamp + config.RECORDING_THUMBNAIL_DELAY

        # Find the buffered frame closest to the target timestamp.
        # Timestamps are monotonic, so we can stop at the first one past target.
        best_frame = self.recording_buffer[0][0]
        min_diff = float("inf")

        for frame, ts in self.recording_buffer:
            diff = abs(ts - target_timestamp)
            if diff < min_diff:
                min_diff = diff
                best_frame = frame
            if ts > target_timestamp:
                break

        out_frame = best_frame.copy()

        # Draw bounding boxes for the snapshot of confirmed tracks
        for track_info in self.thumbnail_tracks:
            x, y, w, h = track_info["bbox"]
            class_name = track_info["class_name"]
            confidence = track_info["confidence"]

            color = (0, 0, 255)  # default: red
            if hasattr(config, "COLORS"):
                color = config.COLORS.get("other", color)
                if class_name == "person":
                    color = config.COLORS.get("person", color)
                elif class_name in ("car", "truck", "bus", "motorcycle", "bicycle"):
                    color = config.COLORS.get("vehicle", color)
                elif class_name in ("cat", "dog"):
                    color = config.COLORS.get("animal", color)

            cv2.rectangle(out_frame, (x, y), (x + w, y + h), color, 2)

            label = f"{class_name} {confidence:.0%}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out_frame, (x, y - text_h - 8), (x + text_w + 4, y), color, -1)
            cv2.putText(
                out_frame, label, (x + 2, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )

        cv2.imwrite(str(path), out_frame)

    def _save_metadata(self, path: Path):
        """Write event metadata as JSON."""
        if not self.current_event:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.current_event.to_dict(), f, indent=2, ensure_ascii=False)

    def _reset_state(self):
        """Reset internal state after a recording is finalized."""
        self.state = RecordingState.IDLE
        self.current_event = None
        self.recording_buffer.clear()
        self.post_buffer_start = 0

    def force_stop(self) -> Optional[str]:
        """
        Force the recorder to flush the current recording (used at shutdown).

        Returns:
            Path to the saved video, if any.
        """
        with self.lock:
            if self.state != RecordingState.IDLE and self.recording_buffer:
                return self._finish_recording(time.time())
        return None

    def get_state(self) -> dict:
        """Return a snapshot of the recorder state."""
        with self.lock:
            return {
                "state": self.state.value,
                "buffer_frames": len(self.recording_buffer),
                "pre_buffer_frames": len(self.pre_buffer),
                "current_event": self.current_event.event_id if self.current_event else None,
                "total_events": self.total_events,
                "total_duration": self.total_duration,
            }

    @property
    def is_recording(self) -> bool:
        return self.state != RecordingState.IDLE
