#!/usr/bin/env python3
"""
Record a test video clip from the camera for the experimental section.

Usage:
    python record_video.py                                # sub-stream, 5 min
    python record_video.py --duration 300                 # 5 minutes
    python record_video.py --duration 600 --stream main   # main-stream, 10 min
    python record_video.py --output my_test.mp4

Output file:
    test_video_<timestamp>_<stream>.mp4  (or whatever --output specifies)

Recommendations for the test clip:
    - Length 5-8 minutes (300-480 seconds)
    - Capture: persons close (~5 m), persons far (~15-18 m), a vehicle, an empty scene
    - Record under the same lighting conditions as the upcoming experiments
    - Use the sub-stream for Exp 1-3 (H.264, more reliable), main for Exp 4

Alternative ffmpeg one-liner:
    # Sub-stream (H.264):
    ffmpeg -rtsp_transport tcp -i "$RTSP_URL" -c copy -t 300 test_video_sub.mp4
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

import config


def parse_args():
    parser = argparse.ArgumentParser(description="Record a test video from the camera")
    parser.add_argument(
        "--duration", type=int, default=300,
        help="Recording duration in seconds (default: 300 = 5 minutes)",
    )
    parser.add_argument(
        "--stream", type=str, default="sub", choices=["sub", "main"],
        help="sub = 896x512 H.264 | main = 4K H.265 (default: sub)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file (default: test_video_<ts>_<stream>.mp4)",
    )
    parser.add_argument(
        "--fps-limit", type=float, default=0,
        help="Cap output FPS (0 = no cap; recommended for main stream: 10)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Build the RTSP URL for the chosen stream
    stream_path = "h264Preview_01_sub" if args.stream == "sub" else "h264Preview_01_main"
    rtsp_url = (
        f"rtsp://{config.CAMERA_USERNAME}:{config.CAMERA_PASSWORD}"
        f"@{config.CAMERA_HOST}:{config.CAMERA_RTSP_PORT}/{stream_path}"
    )

    # Output file
    if args.output:
        output_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"test_video_{ts}_{args.stream}.mp4")

    print("=" * 60)
    print("Test video recording")
    print(f"  Stream:   {args.stream.upper()} ({stream_path})")
    print(f"  Duration: {args.duration}s ({args.duration // 60}m {args.duration % 60}s)")
    print(f"  Output:   {output_path}")
    print("=" * 60)

    print("\nConnecting to the camera...")
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"[ERROR] Cannot connect to {rtsp_url}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    write_fps = args.fps_limit if args.fps_limit > 0 else src_fps

    print(f"[OK] {width}x{height} @ {src_fps:.1f} FPS")
    print(f"     Writing: {write_fps:.1f} FPS")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, write_fps, (width, height))

    if not out.isOpened():
        print(f"[ERROR] Cannot open the output file: {output_path}")
        cap.release()
        sys.exit(1)

    start_time = time.time()
    frame_count = 0
    last_print = start_time
    frame_interval = 1.0 / write_fps if args.fps_limit > 0 else 0
    last_frame_time = 0.0

    print("\nRecording... (Ctrl+C to stop)\n")

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed >= args.duration:
                print(f"\n[OK] Duration reached ({args.duration}s)")
                break

            ret, frame = cap.read()
            if not ret:
                print("\n[WARN] Stream lost, retrying...")
                time.sleep(0.5)
                continue

            # FPS limiter
            now = time.time()
            if frame_interval > 0 and (now - last_frame_time) < frame_interval:
                continue
            last_frame_time = now

            out.write(frame)
            frame_count += 1

            # Print progress every 10 seconds
            if now - last_print >= 10:
                last_print = now
                remaining = args.duration - elapsed
                size_mb = output_path.stat().st_size / 1024 ** 2 if output_path.exists() else 0
                print(
                    f"  [{elapsed:5.0f}s / {args.duration}s]  "
                    f"Remaining: {remaining:.0f}s  |  "
                    f"Frames: {frame_count}  |  "
                    f"Size: {size_mb:.1f} MB"
                )

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n[STOP] Interrupted after {elapsed:.0f}s")

    finally:
        cap.release()
        out.release()

        if output_path.exists():
            size_mb = output_path.stat().st_size / 1024 ** 2
            actual_dur = frame_count / write_fps
            print(f"\n[OK] Saved: {output_path}")
            print(f"     Frames:   {frame_count}")
            print(f"     Duration: {actual_dur:.0f}s")
            print(f"     Size:     {size_mb:.1f} MB")
            print("\nUsing this clip for experiments (config.py):")
            print(f'  INPUT_SOURCE = "./{output_path.name}"')
            print("  PLAYBACK_SPEED = 1.0")
        else:
            print("[ERROR] Output file was not created")


if __name__ == "__main__":
    main()
