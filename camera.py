#!/usr/bin/env python3
"""
Video stream reader - RTSP camera or file source.

Modes:
  RTSP (live stream): source = "rtsp://..."
    - Threaded reader - always returns the most recent frame, non-blocking
    - Callbacks for the recording buffer

  Video file (experiments): source = "./test_video.mp4"
    - Sequential frame-by-frame reading (synchronous with the main loop)
    - Optional playback acceleration (PLAYBACK_SPEED > 1.0)
    - Stops automatically at end of file (is_opened() -> False)
    - Callbacks work the same way as in RTSP mode
"""

import threading
import time

import cv2


class CameraStream:
    """
    Unified video reader supporting both RTSP and file sources.
    The main loop (main.py) does not need to know the source type.
    """

    def __init__(self, source: str, playback_speed: float = 1.0):
        """
        Args:
            source:         RTSP URL or path to a video file.
            playback_speed: 1.0 = real time, 2.0 = 2x faster (file mode only).
        """
        self.source = source
        self.playback_speed = max(0.1, playback_speed)
        self._is_file = not source.startswith("rtsp://")

        self.stream = cv2.VideoCapture(source)
        if not self._is_file:
            # Keep only the most recent frame; avoids latency in live streams
            self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()
        self.callbacks = []

        # File-only: synchronous read in the main loop
        self._frame_interval = 0.0   # seconds between frames during playback
        if self._is_file:
            fps = self.stream.get(cv2.CAP_PROP_FPS) or 25.0
            self._frame_interval = (1.0 / fps) / self.playback_speed
            total = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"   Video file: {source}")
            print(f"   Length: {total} frames @ {fps:.1f} FPS "
                  f"= {total/fps:.0f}s | Speed: {self.playback_speed}x")

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def start(self):
        """Start reading. No-op for file sources (read happens in read())."""
        if not self._is_file:
            thread = threading.Thread(target=self._update, daemon=True)
            thread.start()
        return self

    def stop(self):
        self.stopped = True
        time.sleep(0.1)
        self.stream.release()

    # -------------------------------------------------------------------------
    # RTSP - threaded reader
    # -------------------------------------------------------------------------

    def _update(self):
        """Background thread for RTSP - always stores the latest frame."""
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if grabbed:
                with self.lock:
                    self.frame = frame
                    cbs = list(self.callbacks)
                for cb in cbs:
                    try:
                        cb(frame)
                    except Exception as e:
                        print(f"Error in camera callback: {e}")
            else:
                time.sleep(0.01)

    # -------------------------------------------------------------------------
    # Frame reading
    # -------------------------------------------------------------------------

    def read(self):
        """
        Return the current frame.

        RTSP:  returns the latest frame from the background thread (non-blocking).
        File:  reads the next frame sequentially with playback rate control.
        """
        if self._is_file:
            return self._read_file()

        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def _read_file(self):
        """Sequential read from a file with FPS pacing."""
        t0 = time.perf_counter()

        grabbed, frame = self.stream.read()
        if not grabbed:
            self.stopped = True
            return None

        with self.lock:
            self.frame = frame
            self.grabbed = grabbed
            cbs = list(self.callbacks)

        for cb in cbs:
            try:
                cb(frame)
            except Exception as e:
                print(f"Error in camera callback: {e}")

        # Pace playback to match the requested speed
        elapsed = time.perf_counter() - t0
        wait = self._frame_interval - elapsed
        if wait > 0:
            time.sleep(wait)

        return frame.copy()

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def add_callback(self, callback):
        with self.lock:
            if callback not in self.callbacks:
                self.callbacks.append(callback)

    def remove_callback(self, callback):
        with self.lock:
            if callback in self.callbacks:
                self.callbacks.remove(callback)

    def is_opened(self) -> bool:
        if self._is_file:
            return self.stream.isOpened() and not self.stopped
        return self.stream.isOpened() and self.grabbed

    def is_file_mode(self) -> bool:
        return self._is_file

    def get_properties(self) -> dict:
        return {
            "width":  int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps":    self.stream.get(cv2.CAP_PROP_FPS),
            "source": "file" if self._is_file else "rtsp",
            "total_frames": int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT)) if self._is_file else 0,
        }

    def get_progress(self) -> dict:
        """File playback progress (used for console output)."""
        if not self._is_file:
            return {}
        pos = int(self.stream.get(cv2.CAP_PROP_POS_FRAMES))
        total = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        pct = pos / total * 100 if total > 0 else 0
        fps = self.stream.get(cv2.CAP_PROP_FPS) or 25.0
        return {
            "frame": pos, "total": total,
            "pct": pct,
            "elapsed_s": pos / fps,
            "total_s": total / fps,
        }
