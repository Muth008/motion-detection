#!/usr/bin/env python3
"""
Main module of the motion detection system.

Pipeline:
    1. CameraStream:    read RTSP stream or video file
    2. MotionDetector:  motion detection (MOG2)
    3. ObjectDetector:  object classification (YOLO)
    4. ObjectTracker:   temporal tracking
    5. EventRecorder:   record video clips around detection events
    6. Evaluator:       collect performance metrics for the experimental section

Keyboard shortcuts:
    q     - quit
    d     - toggle debug windows
    b     - show MOG2 background model
    y     - toggle YOLO on/off
    t     - toggle display of tentative tracks
    r     - reset tracker
    space - toggle recording on/off
    e     - print live evaluator stats to console
"""

import sys
import time

import cv2

import config
from camera import CameraStream
from evaluator import Evaluator
from motion_detector import MotionDetector
from object_detector import ObjectDetector
from recorder import EventRecorder
from tracker import ObjectTracker, TrackState


# =============================================================================
# Drawing helpers
# =============================================================================

def get_color_for_class(class_name: str) -> tuple:
    if class_name == "person":
        return config.COLORS["person"]
    if class_name in ("car", "truck", "bus", "motorcycle", "bicycle"):
        return config.COLORS["vehicle"]
    if class_name in ("cat", "dog"):
        return config.COLORS["animal"]
    return config.COLORS["other"]


def draw_motion_regions(frame, regions):
    for r in regions:
        cv2.rectangle(
            frame, (r.x, r.y), (r.x + r.width, r.y + r.height),
            config.COLORS["motion"], 1,
        )


def draw_tracks(frame, tracks: list, show_tentative: bool = False):
    for track in tracks:
        if track.state == TrackState.TENTATIVE and not show_tentative:
            continue
        if track.state == TrackState.LOST:
            continue

        x, y, w, h = track.current_bbox
        color = get_color_for_class(track.class_name)
        thickness = 2 if track.is_confirmed() else 1

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

        if track.is_confirmed():
            label = f"#{track.track_id} {track.class_name} {track.median_confidence:.0%}"
        else:
            label = f"#{track.track_id} {track.class_name} ({track.hits}/{config.TRACKER_MIN_HITS})"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        bg = color if track.is_confirmed() else (128, 128, 128)
        cv2.rectangle(frame, (x, y - th - 8), (x + tw + 4, y), bg, -1)
        cv2.putText(
            frame, label, (x + 2, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )
        cv2.circle(frame, track.center, 4, color, -1)


def draw_recording_indicator(frame, recorder):
    state = recorder.get_state()
    if state["state"] == "idle":
        return
    # Blink red dot at 1 Hz while recording
    if int(time.time() * 2) % 2 == 0:
        cv2.circle(frame, (frame.shape[1] - 30, 30), 15, (0, 0, 255), -1)
    label = "REC" if state["state"] == "recording" else "POST"
    cv2.putText(
        frame, f"{label}: {state['buffer_frames']}f",
        (frame.shape[1] - 160, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
    )


def create_debug_view(debug_info):
    raw = cv2.cvtColor(debug_info["raw_mask"], cv2.COLOR_GRAY2BGR)
    no_shadow = cv2.cvtColor(debug_info["no_shadows_mask"], cv2.COLOR_GRAY2BGR)
    clean = cv2.cvtColor(debug_info["clean_mask"], cv2.COLOR_GRAY2BGR)
    for img, label in [
        (raw, "1. Raw MOG2"),
        (no_shadow, "2. No Shadows"),
        (clean, "3. Morphology"),
    ]:
        cv2.putText(img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    return cv2.hconcat([raw, no_shadow, clean])


# =============================================================================
# Main loop
# =============================================================================

def main():
    print("=" * 60)
    print("Motion Detection System")
    print(f"Model: {config.YOLO_MODEL} | Session: {config.EVAL_SESSION_LABEL}")
    print("=" * 60)

    # --- 1. Camera or video file ----------------------------------------------
    source = getattr(config, "INPUT_SOURCE", None) or config.RTSP_URL
    playback_speed = getattr(config, "PLAYBACK_SPEED", 1.0)
    is_file = not source.startswith("rtsp://")

    if is_file:
        import os
        if not os.path.exists(source):
            print(f"[ERROR] Video file not found: {source}")
            sys.exit(1)
        print(f"\n[1/5] Opening video file (experimental mode)...")
    else:
        print(f"\n[1/5] Connecting to the camera...")

    camera = CameraStream(source, playback_speed=playback_speed)
    if not is_file:
        time.sleep(0.5)
    if not camera.is_opened():
        print("[ERROR] Failed to open the video source.")
        sys.exit(1)
    props = camera.get_properties()
    fps = props["fps"] if 0 < props["fps"] <= 120 else 25.0
    src_label = f"file ({playback_speed}x)" if is_file else "RTSP"
    print(f"[OK] {props['width']}x{props['height']} @ {fps:.1f} FPS [{src_label}]")

    # --- 2. MOG2 --------------------------------------------------------------
    print(f"\n[2/5] Initializing MOG2...")
    motion_detector = MotionDetector()
    print("[OK] MOG2 ready")

    # --- 3. YOLO --------------------------------------------------------------
    print(f"\n[3/5] Loading YOLO ({config.YOLO_MODEL})...")
    try:
        object_detector = ObjectDetector()
        yolo_enabled = True
        print("[OK] YOLO ready")
    except Exception as e:
        print(f"[WARN] YOLO unavailable: {e}")
        object_detector = None
        yolo_enabled = False

    # --- 4. Tracker + Recorder ------------------------------------------------
    print(f"\n[4/5] Initializing tracker and recorder...")
    tracker = ObjectTracker()
    recorder = EventRecorder(fps=fps)
    recording_enabled = config.RECORDING_ENABLED

    # The recording buffer keeps a downscaled HD copy instead of the full 4K
    # frame. This reduces memory usage roughly 6x while the detection pipeline
    # still works with the full-resolution frame.
    rec_width = getattr(config, "RECORDING_BUFFER_WIDTH", 1280)
    rec_height = getattr(config, "RECORDING_BUFFER_HEIGHT", 720)

    def _recorder_callback(frame):
        small = cv2.resize(frame, (rec_width, rec_height))
        recorder.add_frame(small)

    camera.add_callback(_recorder_callback)
    print(f"[OK] Tracker and recorder ready (buffer resize: {rec_width}x{rec_height})")

    # --- 5. Evaluator ---------------------------------------------------------
    print(f"\n[5/5] Starting evaluator...")
    evaluator = (
        Evaluator(session_label=config.EVAL_SESSION_LABEL)
        if config.EVAL_ENABLED else None
    )
    if evaluator:
        evaluator.start()

    # Start asynchronous reading of the live stream (no-op for files)
    camera.start()

    print("\n" + "-" * 60)
    print("Keys: q=quit | d=debug | b=background | y=yolo | t=tentative | "
          "r=reset | space=rec | e=eval stats")
    print("-" * 60 + "\n")

    # --- Loop state -----------------------------------------------------------
    frame_count = 0          # processed frames only (after FRAME_SKIP)
    total_frames_raw = 0     # all received frames (used by FRAME_SKIP logic)
    fps_start = time.time()
    fps_display = 0.0
    show_tentative = config.TRACKER_SHOW_TENTATIVE
    confirmed_events = 0
    reported_track_ids: set = set()  # avoid reporting the same track twice
    last_stats_print = time.time()
    stats_print_interval = getattr(config, "STATS_PRINT_INTERVAL", 30)

    try:
        while True:
            frame = camera.read()
            if frame is None:
                if camera.is_file_mode():
                    print("\n[VIDEO] End of file - session terminated.")
                    break
                continue

            total_frames_raw += 1
            stage_times = {}
            yolo_ran = False

            # FRAME_SKIP: skip processing this frame; the recorder buffer is
            # still updated by the camera-thread callback.
            if hasattr(config, "FRAME_SKIP") and config.FRAME_SKIP > 1:
                if total_frames_raw % config.FRAME_SKIP != 0:
                    elapsed = time.time() - fps_start
                    if elapsed >= 1.0:
                        fps_display = frame_count / elapsed
                        frame_count = 0
                        fps_start = time.time()
                    continue

            frame_count += 1   # counts processed frames only

            # -- STAGE 1: MOG2 -------------------------------------------------
            t0 = time.perf_counter()
            motion_regions, debug_info = motion_detector.detect(frame)
            stage_times["mog2"] = time.perf_counter() - t0

            # -- STAGE 2: YOLO -------------------------------------------------
            raw_detections = []
            stage_times["yolo"] = 0.0
            if yolo_enabled and object_detector is not None:
                should_run = not config.YOLO_ONLY_ON_MOTION or len(motion_regions) > 0
                if should_run:
                    yolo_ran = True
                    t0 = time.perf_counter()
                    if config.YOLO_DETECTION_MODE == "motion_regions" and motion_regions:
                        raw_detections = object_detector.detect_in_motion_regions(
                            frame, motion_regions
                        )
                    else:
                        raw_detections = object_detector.detect(frame)
                    stage_times["yolo"] = time.perf_counter() - t0

            # -- STAGE 3: Tracker ----------------------------------------------
            t0 = time.perf_counter()
            confirmed_tracks = tracker.update(raw_detections)
            all_tracks = tracker.get_all_tracks()
            stage_times["tracker"] = time.perf_counter() - t0

            track_counts = tracker.get_track_count()

            # Log newly confirmed tracks; ``hits >= MIN_HITS`` would otherwise
            # be true on every subsequent frame, so we deduplicate by track id.
            new_confirmations = 0
            for track in confirmed_tracks:
                if (track.track_id not in reported_track_ids
                        and track.hits >= config.TRACKER_MIN_HITS):
                    reported_track_ids.add(track.track_id)
                    new_confirmations += 1
                    confirmed_events += 1
                    print(
                        f"[CONFIRMED] #{track.track_id} {track.class_name} "
                        f"({track.median_confidence:.0%})"
                    )
                    if evaluator:
                        evaluator.increment_event()

            # -- STAGE 4: Recording --------------------------------------------
            if recording_enabled:
                saved_path = recorder.update(confirmed_tracks)
                if saved_path:
                    print(f"[SAVED] {saved_path}")
                    if evaluator:
                        evaluator.increment_recording()
            else:
                if recorder.is_recording:
                    saved = recorder.force_stop()
                    if saved:
                        print(f"[SAVED] {saved}")

            # -- STAGE 5: Evaluator --------------------------------------------
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps_display = frame_count / elapsed
                frame_count = 0
                fps_start = time.time()

            if evaluator:
                evaluator.record_frame(
                    motion_regions=len(motion_regions),
                    yolo_detections=len(raw_detections),
                    confirmed_tracks=len(confirmed_tracks),
                    new_confirmations=new_confirmations,
                    stage_times=stage_times,
                    fps=fps_display,
                    yolo_ran=yolo_ran,
                )

            # -- Periodic console output (helpful for headless RPi) ------------
            now = time.time()
            if now - last_stats_print >= stats_print_interval:
                last_stats_print = now
                live = evaluator.get_live_stats() if evaluator else {}
                rec_state = recorder.get_state()
                # Fall back to psutil if the evaluator does not have CPU/RAM yet
                cpu_str = f"{live.get('avg_cpu_pct', 0):.0f}%"
                ram_str = f"{live.get('avg_ram_mb', 0):.0f}MB"
                if live.get("avg_cpu_pct", 0) == 0:
                    try:
                        import psutil
                        cpu_str = f"{psutil.cpu_percent():.0f}%"
                        ram_str = f"{psutil.Process().memory_info().rss / 1024 ** 2:.0f}MB"
                    except Exception:
                        cpu_str = "n/a"
                progress = camera.get_progress()
                prog_str = (
                    f" | Video: {progress.get('elapsed_s', 0):.0f}s"
                    f"/{progress.get('total_s', 0):.0f}s"
                    f" ({progress.get('pct', 0):.0f}%)"
                ) if progress else ""
                print(
                    f"[STATS] "
                    f"FPS: {fps_display:.1f} | "
                    f"CPU: {cpu_str} | "
                    f"RAM: {ram_str} | "
                    f"YOLO: {live.get('avg_yolo_ms', 0):.0f}ms "
                    f"(p95: {live.get('p95_yolo_ms', 0):.0f}ms) | "
                    f"MOG2: {live.get('avg_mog2_ms', 0):.1f}ms | "
                    f"Motion frames: {live.get('frames_with_motion', 0)} | "
                    f"Events: {confirmed_events} | "
                    f"Recordings: {rec_state['total_events']}"
                    f"{prog_str}"
                )

            # -- Drawing -------------------------------------------------------
            draw_motion_regions(frame, motion_regions)
            draw_tracks(frame, all_tracks, show_tentative=show_tentative)
            if recording_enabled:
                draw_recording_indicator(frame, recorder)

            # Info overlay
            rec_state = recorder.get_state()
            live = evaluator.get_live_stats() if evaluator else {}
            yolo_ms_str = f"{live.get('avg_yolo_ms', 0):.0f}ms" if live else "-"
            far_str = f"{live.get('far_yolo', 0):.0f}%" if live else "-"

            info_lines = [
                f"FPS: {fps_display:.1f} | MOG2: {len(motion_regions)} | "
                f"YOLO: {'ON' if yolo_enabled else 'OFF'} ({yolo_ms_str})",
                f"Tracks: {track_counts['confirmed']} confirmed | Events: {confirmed_events}",
                f"FAR reduction (YOLO): {far_str} | "
                f"Rec: {'ON' if recording_enabled else 'OFF'} [{rec_state['total_events']}]",
            ]
            for i, line in enumerate(info_lines):
                cv2.putText(
                    frame, line, (10, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2,
                )

            if config.SHOW_MAIN_WINDOW:
                cv2.imshow("Detection System", frame)
            if config.SHOW_DEBUG_WINDOWS:
                cv2.imshow("Debug: Masks", create_debug_view(debug_info))

            # -- Keyboard handling --------------------------------------------
            key = cv2.waitKey(config.WINDOW_WAIT_MS) & 0xFF
            if key == ord("q"):
                print("\nShutting down...")
                break
            elif key == ord("d"):
                config.SHOW_DEBUG_WINDOWS = not config.SHOW_DEBUG_WINDOWS
                if not config.SHOW_DEBUG_WINDOWS:
                    cv2.destroyWindow("Debug: Masks")
            elif key == ord("b"):
                bg = motion_detector.get_background()
                if bg is not None:
                    cv2.imshow("Background Model", bg)
            elif key == ord("y"):
                if object_detector:
                    yolo_enabled = not yolo_enabled
                    print(f"YOLO: {'ON' if yolo_enabled else 'OFF'}")
            elif key == ord("t"):
                show_tentative = not show_tentative
                print(f"Show tentative: {'ON' if show_tentative else 'OFF'}")
            elif key == ord("r"):
                tracker.reset()
                print("Tracker reset")
            elif key == ord(" "):
                recording_enabled = not recording_enabled
                if recording_enabled:
                    camera.add_callback(_recorder_callback)
                else:
                    camera.remove_callback(_recorder_callback)
                print(f"Recording: {'ON' if recording_enabled else 'OFF'}")
            elif key == ord("e"):
                if evaluator:
                    stats = evaluator.get_live_stats()
                    print(
                        f"\n[EVAL] Frames: {stats.get('frames')} | "
                        f"FPS: {stats.get('avg_fps')} | "
                        f"MOG2: {stats.get('avg_mog2_ms')}ms | "
                        f"YOLO: {stats.get('avg_yolo_ms')}ms | "
                        f"FAR(YOLO): {stats.get('far_yolo')}%"
                    )

    except KeyboardInterrupt:
        print("\nInterrupted (Ctrl+C)...")

    finally:
        if recorder.is_recording:
            saved = recorder.force_stop()
            if saved:
                print(f"[SAVED] {saved}")

        if evaluator:
            evaluator.stop()

        rec_state = recorder.get_state()
        print(
            f"\nRecorded clips: {rec_state['total_events']} | "
            f"Total duration: {rec_state['total_duration']:.1f}s"
        )

        camera.stop()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
