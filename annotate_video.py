#!/usr/bin/env python3
"""
Ground-truth annotation tool for a test video.

Plays the video and lets the user mark events with the keyboard in real time.
Output: ``ground_truth.csv`` - the basis for TP/FP/FN evaluation.

Usage:
    python annotate_video.py test_video_sub.mp4
    python annotate_video.py test_video_sub.mp4 --output ground_truth.csv
    python annotate_video.py test_video_sub.mp4 --speed 0.5   # slow playback

Keyboard shortcuts during playback:
    1     - person CLOSE (< 8 m)
    2     - person FAR   (> 8 m)
    3     - vehicle
    4     - animal
    5     - other movement (branch, shadow, light)
    SPACE - pause / resume
    LEFT  - jump back 5 s
    RIGHT - jump forward 5 s
    d     - remove the most recent annotation
    s     - save (incremental)
    q     - quit and save

Output file (ground_truth.csv):
    timestamp_s, class_name, distance, video_time, notes
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2


# =============================================================================
# Class and key configuration
# =============================================================================

CLASSES = {
    ord("1"): ("person",  "close", (0, 200, 0),     "Person CLOSE  (<8m)"),
    ord("2"): ("person",  "far",   (0, 100, 255),   "Person FAR    (>8m)"),
    ord("3"): ("vehicle", "close", (0, 165, 255),   "Vehicle"),
    ord("4"): ("animal",  "close", (255, 0, 255),   "Animal"),
    ord("5"): ("other",   "close", (128, 128, 128), "Other movement (FP source)"),
}


def parse_args():
    p = argparse.ArgumentParser(description="Ground-truth annotation tool")
    p.add_argument("video", type=str, help="Path to the video file")
    p.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output CSV file (default: ground_truth_<video>.csv)",
    )
    p.add_argument(
        "--speed", "-s", type=float, default=1.0,
        help="Playback speed (0.5 = slower, 1.0 = normal)",
    )
    return p.parse_args()


def fmt_time(seconds: float) -> str:
    m = int(seconds) // 60
    s = seconds % 60
    return f"{m:02d}:{s:05.2f}"


def draw_overlay(frame, annotations, paused, current_s, total_s, speed):
    h, w = frame.shape[:2]

    # Progress bar
    bar_x, bar_y, bar_w, bar_h = 20, h - 30, w - 40, 12
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    progress = int(bar_w * current_s / total_s) if total_s > 0 else 0
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress, bar_y + bar_h), (0, 200, 100), -1)

    # Time
    time_str = f"{fmt_time(current_s)} / {fmt_time(total_s)}"
    cv2.putText(frame, time_str, (bar_x, bar_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Pause / speed
    status = f"[PAUSED]  {speed}x" if paused else f">  {speed}x"
    cv2.putText(frame, status, (w - 160, bar_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 255) if paused else (200, 200, 200), 1)

    # Key legend
    legend = [
        "1=Person close  2=Person far",
        "3=Vehicle  4=Animal  5=Other motion",
        "SPACE=pause  LEFT/RIGHT=+-5s",
        "d=remove last  q=save+quit",
    ]
    for i, line in enumerate(legend):
        cv2.putText(frame, line, (20, 25 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # Last 3 annotations
    cv2.putText(frame, f"Annotations: {len(annotations)}", (w - 200, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 100), 1)
    for i, ann in enumerate(annotations[-3:]):
        color = CLASSES.get(ann["_key"], (None,) * 4)[2] or (200, 200, 200)
        cv2.putText(
            frame,
            f"  {ann['video_time']}  {ann['class_name']} {ann['distance']}",
            (w - 280, 50 + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
        )

    # Flash on new annotation
    if annotations and (time.time() - annotations[-1].get("_added_at", 0)) < 0.4:
        color = CLASSES.get(annotations[-1]["_key"], (None,) * 4)[2] or (0, 255, 0)
        label = f"OK  {annotations[-1]['class_name'].upper()}  {annotations[-1]['distance']}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cx = (w - tw) // 2
        cv2.rectangle(frame, (cx - 10, h // 2 - 40), (cx + tw + 10, h // 2 + 15), (0, 0, 0), -1)
        cv2.putText(frame, label, (cx, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    return frame


def save_csv(output_path, annotations):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["timestamp_s", "class_name", "distance", "video_time", "notes"]
        )
        writer.writeheader()
        for ann in annotations:
            writer.writerow({
                "timestamp_s": round(ann["timestamp_s"], 3),
                "class_name":  ann["class_name"],
                "distance":    ann["distance"],
                "video_time":  ann["video_time"],
                "notes":       ann.get("notes", ""),
            })
    return len(annotations)


def main():
    args = parse_args()
    video_path = Path(args.video)

    if not video_path.exists():
        print(f"[ERROR] File not found: {video_path}")
        sys.exit(1)

    output_path = (
        Path(args.output) if args.output
        else video_path.parent / f"ground_truth_{video_path.stem}.csv"
    )

    # Load existing annotations if the file already exists (resume)
    annotations = []
    if output_path.exists():
        with open(output_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["timestamp_s"] = float(row["timestamp_s"])
                row["_key"] = 0
                row["_added_at"] = 0
                annotations.append(row)
        print(f"[LOAD] {len(annotations)} existing annotations from {output_path.name}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_s = total_f / fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Shrink the window if the video is too big
    display_scale = min(1.0, 1280 / w, 800 / h)
    disp_w = int(w * display_scale)
    disp_h = int(h * display_scale)

    print(f"\n[OK] {video_path.name}  |  {w}x{h}  |  {fps:.1f} FPS  |  {fmt_time(total_s)}")
    print(f"     Output:  {output_path}")
    print(f"     Display: {disp_w}x{disp_h} (scale {display_scale:.2f})")
    print("\nStarting annotation...\n")

    cv2.namedWindow("Annotator", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Annotator", disp_w, disp_h)

    paused = False
    frame_interval_ms = int(1000 / (fps * args.speed))
    frame = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\n[OK] End of video")
                break

        if frame is None:
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            continue

        current_f = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        current_s = current_f / fps

        display = frame.copy()
        if display_scale < 1.0:
            display = cv2.resize(display, (disp_w, disp_h))

        draw_overlay(display, annotations, paused, current_s, total_s, args.speed)
        cv2.imshow("Annotator", display)

        key = cv2.waitKey(frame_interval_ms if not paused else 30) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == 81 or key == 2424832:    # LEFT arrow
            new_f = max(0, current_f - int(fps * 5))
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_f)
            ret, frame = cap.read()
        elif key == 83 or key == 2555904:    # RIGHT arrow
            new_f = min(total_f - 1, current_f + int(fps * 5))
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_f)
            ret, frame = cap.read()
        elif key == ord("d"):
            if annotations:
                removed = annotations.pop()
                print(
                    f"[REMOVE] {removed['video_time']} "
                    f"{removed['class_name']} {removed['distance']}"
                )
        elif key == ord("s"):
            n = save_csv(output_path, annotations)
            print(f"[SAVE] Wrote {n} annotations -> {output_path}")
        elif key in CLASSES:
            class_name, distance, _, label = CLASSES[key]
            ann = {
                "timestamp_s": current_s,
                "class_name":  class_name,
                "distance":    distance,
                "video_time":  fmt_time(current_s),
                "notes":       "",
                "_key":        key,
                "_added_at":   time.time(),
            }
            annotations.append(ann)
            print(f"[ADD]  {fmt_time(current_s)}  {class_name} {distance}")

    cap.release()
    cv2.destroyAllWindows()

    if annotations:
        n = save_csv(output_path, annotations)
        print(f"\n[SAVE] Wrote {n} annotations -> {output_path}")
        print("\nAnnotation preview:")
        print(f"  {'Time':>8}  {'Class':10}  {'Distance':10}")
        print(f"  {'-' * 32}")
        for ann in annotations:
            print(f"  {ann['video_time']:>8}  {ann['class_name']:10}  {ann['distance']}")
    else:
        print("\n[WARN] No annotations were saved")


if __name__ == "__main__":
    main()
