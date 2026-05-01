#!/usr/bin/env python3
"""
Compare experiment results to a ground-truth annotation file.

Two evaluation modes are supported:
    1. Recall experiment - GT contains real persons; we measure
       TP / FP / FN / Precision / Recall / F1.
    2. FAR experiment    - GT contains "other" events (branches, shadows...);
       we measure False Alarm Rate.

Usage:
    # Recall (Exp 1-5):
    python evaluate_experiment.py ground_truth_sub.csv --results eval_results/

    # FAR (Exp 7):
    python evaluate_experiment.py ground_truth_far_branch.csv \\
        --results eval_results/ --sessions far_branch_sys far_branch_reolink

    # Combined (Scenario C - branch + person):
    python evaluate_experiment.py ground_truth_far_combined.csv --results eval_results/

    # Filter sessions:
    python evaluate_experiment.py gt.csv --sessions yolo11n_exp1 yolo11s_exp1
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_TOLERANCE = 5.0


def parse_args():
    p = argparse.ArgumentParser(description="Compare experiments against ground truth")
    p.add_argument("ground_truth", type=str, help="ground_truth CSV file")
    p.add_argument(
        "--results", "-r", type=str, default="eval_results",
        help="Directory containing eval CSV files (default: eval_results/)",
    )
    p.add_argument(
        "--tolerance", "-t", type=float, default=DEFAULT_TOLERANCE,
        help=f"Matching tolerance in seconds (default: {DEFAULT_TOLERANCE})",
    )
    p.add_argument(
        "--sessions", "-s", nargs="+", default=None,
        help="Filter to only the given session labels",
    )
    p.add_argument(
        "--source-fps", "-f", type=float, default=None,
        help="Source video FPS for frame->time conversion. None = auto-detect from GT",
    )
    p.add_argument(
        "--output", "-o", type=str, default="experiment_results",
        help="Output filename prefix (default: experiment_results)",
    )
    return p.parse_args()


# =============================================================================
# Data loading
# =============================================================================

def load_ground_truth(path: str) -> list:
    events = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            events.append({
                "timestamp_s": float(row["timestamp_s"]),
                "class_name":  row["class_name"],
                "distance":    row["distance"],
                "video_time":  row["video_time"],
                "notes":       row.get("notes", ""),
            })
    events.sort(key=lambda x: x["timestamp_s"])
    return events


def load_eval_sessions(results_dir: str, session_filter: list = None) -> dict:
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"[ERROR] Directory not found: {results_dir}")
        sys.exit(1)

    sessions = defaultdict(list)
    for csv_file in sorted(results_path.glob("*.csv")):
        # Skip output files and Reolink collector data
        if any(x in csv_file.name for x in ["reolink", "experiment_results", "ground_truth"]):
            continue

        # Strip the leading timestamp from the filename to recover the session label
        stem = csv_file.stem
        parts = stem.split("_")
        label_parts = []
        skip = True
        for part in parts:
            if skip and part.isdigit():
                continue
            skip = False
            label_parts.append(part)
        session_label = "_".join(label_parts) if label_parts else stem

        if session_filter and session_label not in session_filter:
            continue

        with open(csv_file, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            if rows:
                sessions[session_label].extend(rows)
                print(f"   [LOAD] {csv_file.name}  ({len(rows)} frames) -> {session_label}")

    return dict(sessions)


# =============================================================================
# Extract YOLO events from per-frame CSV
# =============================================================================

def extract_yolo_events(frames: list, source_fps: float = None) -> list:
    try:
        frames_sorted = sorted(frames, key=lambda r: int(r.get("frame_index", 0)))
    except Exception:
        return []

    ref_ts = float(frames_sorted[0].get("timestamp", 0)) if frames_sorted else 0
    events = []

    for row in frames_sorted:
        if source_fps and source_fps > 0:
            ts = int(row.get("frame_index", 0)) / source_fps
        else:
            ts = float(row.get("timestamp", 0)) - ref_ts

        new_conf = int(row.get("new_confirmations", 0))
        if new_conf > 0:
            for _ in range(new_conf):
                events.append({
                    "timestamp_s": ts,
                    "class_name":  "person",
                    "track_id":    f"t{len(events)}",
                })

    return events


# =============================================================================
# Match GT <-> YOLO events
# =============================================================================

def match_events(gt_events: list, yolo_events: list, tolerance: float) -> dict:
    """
    Match ground-truth events with YOLO events within a tolerance window.

    Only GT classes person/vehicle/animal are matched as TP - 'other' GT events
    are always treated as a source of false positives.
    """
    real_gt = [e for e in gt_events if e["class_name"] != "other"]
    other_gt = [e for e in gt_events if e["class_name"] == "other"]

    gt_used = [False] * len(real_gt)
    yolo_used = [False] * len(yolo_events)
    tp_pairs = []

    for gi, gt in enumerate(real_gt):
        best_dist = float("inf")
        best_yi = -1
        for yi, yolo in enumerate(yolo_events):
            if yolo_used[yi]:
                continue
            dist = abs(yolo["timestamp_s"] - gt["timestamp_s"])
            if dist <= tolerance and dist < best_dist:
                best_dist = dist
                best_yi = yi
        if best_yi >= 0:
            tp_pairs.append((gt, yolo_events[best_yi]))
            gt_used[gi] = True
            yolo_used[best_yi] = True

    fn_list = [gt for i, gt in enumerate(real_gt) if not gt_used[i]]
    fp_list = [yolo for i, yolo in enumerate(yolo_events) if not yolo_used[i]]

    return {
        "tp": tp_pairs,
        "fp": fp_list,
        "fn": fn_list,
        "other_gt": other_gt,
    }


# =============================================================================
# FAR (False Alarm Rate) computation
# =============================================================================

def compute_far(fp_list: list, yolo_events: list, frames: list,
                source_fps: float, other_gt: list, tolerance: float) -> dict:
    """
    False Alarm Rate metrics.

    FAR_total: all FPs / video duration in minutes.
    FAR_other: FPs near a GT 'other' event / video duration in minutes
               -> FPs caused by a specific spurious stimulus (branch, shadow...).
    """
    if not frames:
        return {}

    try:
        frames_sorted = sorted(frames, key=lambda r: int(r.get("frame_index", 0)))
    except Exception:
        return {}

    if source_fps and source_fps > 0:
        total_frames = int(frames_sorted[-1].get("frame_index", len(frames)))
        duration_s = total_frames / source_fps
    else:
        ref_ts = float(frames_sorted[0].get("timestamp", 0))
        last_ts = float(frames_sorted[-1].get("timestamp", 0))
        duration_s = last_ts - ref_ts

    duration_min = duration_s / 60.0 if duration_s > 0 else 1.0

    # FPs near a GT 'other' event
    fp_near_other = 0
    for fp in fp_list:
        for other in other_gt:
            if abs(fp["timestamp_s"] - other["timestamp_s"]) <= tolerance:
                fp_near_other += 1
                break

    far_total = len(fp_list) / duration_min
    far_other = fp_near_other / duration_min

    return {
        "duration_s":    round(duration_s, 1),
        "duration_min":  round(duration_min, 2),
        "total_events":  len(yolo_events),
        "fp_total":      len(fp_list),
        "fp_near_other": fp_near_other,
        "far_total":     round(far_total, 2),   # total FPs/min
        "far_other":     round(far_other, 2),   # FPs/min caused by 'other' stimuli
    }


# =============================================================================
# Performance metrics
# =============================================================================

def compute_perf_metrics(frames: list) -> dict:
    def safe_float(v, default=0.0):
        try:
            return float(v)
        except Exception:
            return default

    yolo_times = [safe_float(r.get("time_yolo_ms")) for r in frames
                  if safe_float(r.get("time_yolo_ms")) > 0]
    fps_vals = [safe_float(r.get("fps")) for r in frames
                if safe_float(r.get("fps")) > 0]
    cpu_vals = [safe_float(r.get("cpu_percent")) for r in frames
                if safe_float(r.get("cpu_percent")) > 0]
    ram_vals = [safe_float(r.get("ram_mb")) for r in frames
                if safe_float(r.get("ram_mb")) > 0]

    def avg(lst):
        return round(sum(lst) / len(lst), 2) if lst else 0.0

    def p95(lst):
        if not lst:
            return 0.0
        s = sorted(lst)
        return round(s[min(int(len(s) * 0.95), len(s) - 1)], 2)

    return {
        "avg_yolo_ms":  avg(yolo_times),
        "p95_yolo_ms":  p95(yolo_times),
        "avg_fps":      avg(fps_vals),
        "avg_cpu_pct":  avg(cpu_vals),
        "avg_ram_mb":   avg(ram_vals),
        "total_frames": len(frames),
    }


# =============================================================================
# Main per-session evaluation
# =============================================================================

def evaluate_session(session_label, frames, gt_events, tolerance, source_fps=None):
    real_gt = [e for e in gt_events if e["class_name"] != "other"]
    has_real_gt = len(real_gt) > 0

    print(f"\n  Session: {session_label}  ({len(frames)} frames)")

    yolo_events = extract_yolo_events(frames, source_fps=source_fps)
    print(f"    YOLO events:    {len(yolo_events)}")
    print(f"    GT persons:     {len(real_gt)}  |  GT 'other': {len(gt_events) - len(real_gt)}")

    matches = match_events(gt_events, yolo_events, tolerance)
    tp = len(matches["tp"])
    fp = len(matches["fp"])
    fn = len(matches["fn"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if tp == 0 and fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0 if not real_gt else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    perf = compute_perf_metrics(frames)
    far = compute_far(matches["fp"], yolo_events, frames,
                      source_fps, matches["other_gt"], tolerance)

    if has_real_gt:
        print(f"    TP={tp}  FP={fp}  FN={fn}  |  "
              f"P={precision * 100:.0f}%  R={recall * 100:.0f}%  F1={f1 * 100:.0f}%")
    else:
        print("    FAR-only test - no real-person GT")
        print(f"    Total FPs (false alarms): {fp}")

    print(f"    FAR: {far.get('far_total', 0):.2f} FP/min  "
          f"(of which 'other' stimuli: {far.get('far_other', 0):.2f} FP/min)  "
          f"| Duration: {far.get('duration_min', 0):.1f} min")
    print(f"    YOLO: {perf['avg_yolo_ms']}ms avg / {perf['p95_yolo_ms']}ms p95  "
          f"| FPS: {perf['avg_fps']}  CPU: {perf['avg_cpu_pct']}%  RAM: {perf['avg_ram_mb']}MB")

    if matches["fn"]:
        print("    FN (missed):")
        for e in matches["fn"]:
            print(f"      {e['video_time']:>8}  {e['class_name']} {e['distance']}")
    if matches["fp"]:
        print("    FP (false alarms):")
        for e in matches["fp"]:
            print(f"      {e['timestamp_s']:8.1f}s  {e['class_name']}")

    result = {
        "session":      session_label,
        "gt_persons":   len(real_gt),
        "gt_other":     len(gt_events) - len(real_gt),
        "yolo_events":  len(yolo_events),
        "tp": tp, "fp": fp, "fn": fn,
        "precision":    round(precision * 100, 1),
        "recall":       round(recall * 100, 1),
        "f1":           round(f1 * 100, 1),
        "far_total":    far.get("far_total", 0),
        "far_other":    far.get("far_other", 0),
        "duration_min": far.get("duration_min", 0),
        **perf,
        "_tp_pairs": [{"gt_time": g["video_time"], "yolo_time": f"{y['timestamp_s']:.1f}s",
                       "class": g["class_name"], "distance": g["distance"]}
                      for g, y in matches["tp"]],
        "_fn_list":  [{"gt_time": g["video_time"], "class": g["class_name"],
                       "distance": g["distance"]} for g in matches["fn"]],
        "_fp_list":  [{"time": f"{y['timestamp_s']:.1f}s"} for y in matches["fp"]],
    }

    return result


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    print("=" * 65)
    print("Experiment evaluation")
    print("=" * 65)

    print(f"\nLoading ground truth: {args.ground_truth}")
    gt_events = load_ground_truth(args.ground_truth)
    real_gt = [e for e in gt_events if e["class_name"] != "other"]
    other_gt = [e for e in gt_events if e["class_name"] == "other"]
    print(f"  {len(real_gt)} persons/vehicles  +  {len(other_gt)} 'other' (FP sources)")
    for e in gt_events:
        tag = "[GT ]" if e["class_name"] != "other" else "[OTH]"
        print(f"    {tag} {e['video_time']:>8}  {e['class_name']:10}  {e['distance']}")

    print(f"\nLoading session files from: {args.results}")
    sessions = load_eval_sessions(args.results, args.sessions)

    if not sessions:
        print("[ERROR] No session files found")
        sys.exit(1)

    # Auto-detect source FPS
    source_fps = args.source_fps
    if source_fps is None and gt_events:
        first_session = next(iter(sessions.values()))
        total_frames = len(first_session)
        max_gt_ts = max(e["timestamp_s"] for e in gt_events)
        if max_gt_ts > 0:
            source_fps = total_frames / max_gt_ts
            print(f"\nAuto-detected source FPS: {source_fps:.2f} "
                  f"({total_frames} frames / {max_gt_ts:.1f}s GT)")

    # Fallback when GT is empty - estimate FPS from wall-clock timestamps
    if source_fps is None:
        first_session = next(iter(sessions.values()))
        if len(first_session) >= 2:
            ts_first = float(first_session[0].get("timestamp", 0))
            ts_last = float(first_session[-1].get("timestamp", 0))
            duration = ts_last - ts_first
            source_fps = len(first_session) / duration if duration > 0 else 10.0
            print(f"Source FPS from wall-clock: {source_fps:.2f} (empty GT)")
        else:
            source_fps = 10.0

    fps_str = f"{source_fps:.2f}" if source_fps else "auto"
    print(f"\n{'=' * 65}")
    print(f"Matching tolerance +-{args.tolerance}s | Source FPS: {fps_str}")
    print(f"{'=' * 65}")

    all_results = []
    for label in sorted(sessions.keys()):
        result = evaluate_session(label, sessions[label], gt_events,
                                  args.tolerance, source_fps=source_fps)
        all_results.append(result)

    # -------------------------------------------------------------------------
    # Output files
    # -------------------------------------------------------------------------
    csv_fields = ["session", "gt_persons", "gt_other", "yolo_events",
                  "tp", "fp", "fn", "precision", "recall", "f1",
                  "far_total", "far_other", "duration_min",
                  "avg_yolo_ms", "p95_yolo_ms", "avg_fps",
                  "avg_cpu_pct", "avg_ram_mb", "total_frames"]

    csv_path = Path(f"{args.output}.csv")
    json_path = Path(f"{args.output}.json")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_results)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "ground_truth_file": args.ground_truth,
            "tolerance_s":       args.tolerance,
            "source_fps":        source_fps,
            "gt_events":         gt_events,
            "sessions":          all_results,
        }, f, ensure_ascii=False, indent=2, default=str)

    # Summary table
    has_persons = any(r["gt_persons"] > 0 for r in all_results)
    has_far = any(r["gt_other"] > 0 or r["fp"] > 0 for r in all_results)

    print(f"\n{'=' * 65}")
    print(f"{'SUMMARY':^65}")
    print(f"{'=' * 65}")

    if has_persons:
        print(f"{'Session':<30} {'P%':>4} {'R%':>4} {'F1%':>4} "
              f"{'TP':>3} {'FP':>3} {'FN':>3} {'FAR/min':>7} {'YOLO':>6} {'FPS':>5}")
        print("-" * 65)
        for r in all_results:
            print(f"{r['session']:<30} {r['precision']:>4.0f} {r['recall']:>4.0f} "
                  f"{r['f1']:>4.0f} {r['tp']:>3} {r['fp']:>3} {r['fn']:>3} "
                  f"{r['far_total']:>7.2f} {r['avg_yolo_ms']:>5.0f}ms {r['avg_fps']:>4.1f}")
    else:
        print(f"{'Session':<30} {'FAR/min':>7} {'FP total':>9} "
              f"{'Duration':>9} {'YOLO':>6} {'FPS':>5}")
        print("-" * 65)
        for r in all_results:
            print(f"{r['session']:<30} {r['far_total']:>7.2f} {r['fp']:>9} "
                  f"{r['duration_min']:>7.1f}m {r['avg_yolo_ms']:>5.0f}ms {r['avg_fps']:>4.1f}")

    print(f"\n[CSV]  {csv_path}")
    print(f"[JSON] {json_path}")


if __name__ == "__main__":
    main()
