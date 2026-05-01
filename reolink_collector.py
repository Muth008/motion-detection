#!/usr/bin/env python3
"""
Reolink P340 AI event collector.

Purpose in the thesis context:
    Section 4.5 - Comparison with the Reolink built-in AI.

Connection notes:
    The Reolink P340 requires HTTPS with a self-signed certificate (port 443).
    For this reason ``_post()`` uses ``ssl.create_default_context()`` with
    ``CERT_NONE``.

Usage:
    python reolink_collector.py              # standalone (start after main.py)
    python reolink_collector.py my_session
"""

import csv
import json
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import config
from evaluator import SessionSync


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class ReoAiEvent:
    """A single AI detection event reported by the Reolink camera."""
    event_id: str
    start_time: float
    end_time: float
    duration_s: float
    class_name: str
    start_iso: str = ""
    end_iso: str = ""


@dataclass
class ReoSummary:
    """Summary statistics for a Reolink AI collection session."""
    session_label: str
    camera_host: str
    start_time: str
    end_time: str = ""
    duration_s: float = 0.0
    total_polls: int = 0
    failed_polls: int = 0
    poll_success_rate_pct: float = 0.0
    sync_start_time: str = ""       # shared start from the sync file (= Evaluator's start)
    events_person: int = 0
    events_vehicle: int = 0
    events_animal: int = 0
    events_motion: int = 0
    events_total: int = 0
    avg_duration_person_s: float = 0.0
    avg_duration_vehicle_s: float = 0.0
    avg_duration_animal_s: float = 0.0


# =============================================================================
# Reolink API client
# =============================================================================

class ReolinkApi:
    """
    Minimal HTTP client for the Reolink camera API.

    The Reolink P340 requires HTTPS with a self-signed certificate, so
    SSL verification is intentionally disabled (CERT_NONE).
    """

    def __init__(self, host: str, username: str, password: str,
                 port: int = 443, timeout: int = 3):
        self.host = host
        self.username = username
        self.password = password
        self.port = port
        self.timeout = timeout
        scheme = "https" if port == 443 else "http"
        self._base_url = f"{scheme}://{host}:{port}/api.cgi"

    def _post(self, commands: list) -> Optional[list]:
        cmd_param = commands[0].get("cmd", "") if commands else ""
        url = (
            f"{self._base_url}?cmd={cmd_param}"
            f"&user={self.username}&password={self.password}"
        )
        payload = json.dumps(commands).encode("utf-8")

        try:
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

            with urllib.request.urlopen(req, timeout=self.timeout, context=ctx) as resp:
                raw_response = resp.read().decode("utf-8")
                return json.loads(raw_response)
        except urllib.error.HTTPError as e:
            print(f"   [DEBUG] HTTP {e.code}: {e.read().decode('utf-8', errors='ignore')}")
            return None
        except urllib.error.URLError as e:
            print(f"   [DEBUG] URLError: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"   [DEBUG] JSON error: {e}")
            return None
        except Exception as e:
            print(f"   [DEBUG] {type(e).__name__}: {e}")
            return None

    def get_ai_state(self, channel: int = 0) -> Optional[dict]:
        result = self._post([
            {"cmd": "GetAiState", "action": 0, "param": {"channel": channel}}
        ])
        if result and result[0].get("code") == 0:
            return result[0].get("value", {})
        return None

    def get_md_state(self, channel: int = 0) -> Optional[int]:
        result = self._post([
            {"cmd": "GetMdState", "action": 0, "param": {"channel": channel}}
        ])
        if result and result[0].get("code") == 0:
            return result[0].get("value", {}).get("state")
        return None

    def test_connection(self) -> bool:
        result = self._post([
            {"cmd": "GetMdState", "action": 0, "param": {"channel": 0}}
        ])
        if result is None:
            return False
        code = result[0].get("code") if result else None
        if code != 0:
            detail = result[0].get("error", {}).get("detail", f"code {code}") if result else "?"
            print(f"   [DEBUG] API error: {detail}")
            return False
        return True


# =============================================================================
# Event collector
# =============================================================================

class ReolinkCollector:
    """Collects AI detection events from the Reolink P340; logs to CSV + JSON."""

    AI_CLASS_MAP = {
        "people":  "person",
        "vehicle": "vehicle",
        "dog_cat": "animal",
        "face":    "person",
    }

    def __init__(self, session_label: str = ""):
        self.session_label = session_label or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(config.EVAL_OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.output_dir / f"reolink_{ts}_{self.session_label}.csv"
        self.json_path = self.output_dir / f"reolink_{ts}_{self.session_label}_summary.json"

        self.api = ReolinkApi(
            host=config.REOLINK_HOST,
            username=config.REOLINK_USERNAME,
            password=config.REOLINK_PASSWORD,
            port=config.REOLINK_PORT,
        )

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._start_time = 0.0
        self._active_events: dict = {}
        self._completed_events: list = []
        self._events_lock = threading.Lock()
        self._total_polls = 0
        self._failed_polls = 0
        self._sync_start_time = ""
        self._csv_file = None
        self._csv_writer = None

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def start(self) -> bool:
        # Sync file - read the reference start time from the evaluator
        sync = SessionSync.read(self.output_dir)
        if sync:
            lag = time.time() - sync.get("start_timestamp", time.time())
            print(f"   [SYNC] session '{sync.get('session_label')}' started {lag:.0f}s ago")
            self._sync_start_time = sync.get("start_time", "")
        else:
            print("   [WARN] Sync file not found - start main.py first")
            self._sync_start_time = ""

        print(f"   Testing Reolink connection ({config.REOLINK_HOST})...")
        if not self.api.test_connection():
            print("   [ERROR] Cannot connect")
            return False
        print("   [OK] Reolink API reachable")

        self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(
            self._csv_file,
            fieldnames=["event_id", "start_iso", "end_iso", "start_time",
                        "end_time", "duration_s", "class_name"],
        )
        self._csv_writer.writeheader()

        self._start_time = time.time()
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

        print(
            f"   [COLLECTOR] started (interval: {config.REOLINK_POLL_INTERVAL}s) | "
            f"CSV: {self.csv_path.name}"
        )
        return True

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

        # Close any events still active at shutdown
        now = time.time()
        for class_name, start_ts in list(self._active_events.items()):
            self._close_event(class_name, start_ts, now)

        if self._csv_file:
            self._csv_file.close()

        summary = self._compute_summary()
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(summary), f, indent=2, ensure_ascii=False)

        print("\n[COLLECTOR] Reolink session terminated")
        print(
            f"   Events: {summary.events_total} "
            f"(persons: {summary.events_person}, "
            f"vehicles: {summary.events_vehicle}, "
            f"animals: {summary.events_animal})"
        )
        print(f"   Polling success rate: {summary.poll_success_rate_pct:.1f}%")
        print(f"   JSON: {self.json_path}")
        return summary

    # -------------------------------------------------------------------------
    # Polling
    # -------------------------------------------------------------------------

    def _poll_loop(self):
        while self._running:
            t0 = time.time()
            self._total_polls += 1

            ai_state = self.api.get_ai_state(channel=config.REOLINK_CHANNEL)
            md_state = self.api.get_md_state(channel=config.REOLINK_CHANNEL)

            if ai_state is None and md_state is None:
                self._failed_polls += 1
            else:
                now = time.time()
                self._process_ai_state(ai_state or {}, now)
                self._process_md_state(md_state, now)

            # Pace polling
            time.sleep(max(0, config.REOLINK_POLL_INTERVAL - (time.time() - t0)))

    def _process_ai_state(self, ai_state: dict, now: float):
        for api_key, class_name in self.AI_CLASS_MAP.items():
            entry = ai_state.get(api_key, {})
            if not isinstance(entry, dict) or entry.get("support", 0) == 0:
                continue
            is_active = entry.get("alarm_state", 0) == 1
            if is_active and class_name not in self._active_events:
                self._active_events[class_name] = now
                print(
                    f"[REOLINK] -> {class_name} "
                    f"({datetime.fromtimestamp(now).strftime('%H:%M:%S')})"
                )
            elif not is_active and class_name in self._active_events:
                self._close_event(class_name, self._active_events.pop(class_name), now)

    def _process_md_state(self, md_state: Optional[int], now: float):
        if md_state is None:
            return
        key = "_motion"
        if md_state == 1 and key not in self._active_events:
            self._active_events[key] = now
        elif md_state == 0 and key in self._active_events:
            self._close_event("motion", self._active_events.pop(key), now)

    def _close_event(self, class_name: str, start_ts: float, end_ts: float):
        event = ReoAiEvent(
            event_id=datetime.fromtimestamp(start_ts).strftime("%Y%m%d_%H%M%S_%f")[:20],
            start_time=start_ts,
            end_time=end_ts,
            duration_s=round(end_ts - start_ts, 2),
            class_name=class_name,
            start_iso=datetime.fromtimestamp(start_ts).isoformat(),
            end_iso=datetime.fromtimestamp(end_ts).isoformat(),
        )
        print(f"[REOLINK] -- {class_name} ended ({event.duration_s:.1f}s)")
        with self._events_lock:
            self._completed_events.append(event)
        if self._csv_writer:
            self._csv_writer.writerow(asdict(event))
            self._csv_file.flush()

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def _compute_summary(self) -> ReoSummary:
        with self._events_lock:
            events = list(self._completed_events)

        def avg_dur(cls):
            d = [e.duration_s for e in events if e.class_name == cls]
            return round(sum(d) / len(d), 2) if d else 0.0

        return ReoSummary(
            session_label=self.session_label,
            camera_host=config.REOLINK_HOST,
            start_time=datetime.fromtimestamp(self._start_time).isoformat(),
            sync_start_time=self._sync_start_time,
            end_time=datetime.now().isoformat(),
            duration_s=round(time.time() - self._start_time, 1),
            total_polls=self._total_polls,
            failed_polls=self._failed_polls,
            poll_success_rate_pct=round(
                (1 - self._failed_polls / max(self._total_polls, 1)) * 100, 1
            ),
            events_total=len(events),
            events_person=sum(1 for e in events if e.class_name == "person"),
            events_vehicle=sum(1 for e in events if e.class_name == "vehicle"),
            events_animal=sum(1 for e in events if e.class_name == "animal"),
            events_motion=sum(1 for e in events if e.class_name == "motion"),
            avg_duration_person_s=avg_dur("person"),
            avg_duration_vehicle_s=avg_dur("vehicle"),
            avg_duration_animal_s=avg_dur("animal"),
        )

    def get_event_count(self) -> dict:
        with self._events_lock:
            events = self._completed_events
        return {
            "total":   len(events),
            "person":  sum(1 for e in events if e.class_name == "person"),
            "vehicle": sum(1 for e in events if e.class_name == "vehicle"),
            "animal":  sum(1 for e in events if e.class_name == "animal"),
            "active":  list(self._active_events.keys()),
        }


# =============================================================================
# Standalone entry point
# =============================================================================

if __name__ == "__main__":
    label = f"reolink_{sys.argv[1] if len(sys.argv) > 1 else config.EVAL_SESSION_LABEL}"

    print("=" * 60)
    print(f"Reolink Collector | Session: {label} | Camera: {config.REOLINK_HOST}")
    print("=" * 60)

    collector = ReolinkCollector(session_label=label)
    if not collector.start():
        sys.exit(1)

    print("\nCollecting data. Press Ctrl+C to stop.\n")
    try:
        while True:
            time.sleep(10)
            c = collector.get_event_count()
            print(
                f"[STATS] Total: {c['total']} | Persons: {c['person']} | "
                f"Vehicles: {c['vehicle']} | Animals: {c['animal']} | "
                f"Active: {c['active']}"
            )
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        collector.stop()
