#!/usr/bin/env python3
"""
Temporal tracker that links detections across consecutive frames into tracks.

Each track moves through a state machine:

    TENTATIVE -> CONFIRMED -> LOST -> (deleted)

A track is created in TENTATIVE state on the first detection. Once it
accumulates ``TRACKER_MIN_HITS`` matches it is promoted to CONFIRMED, which
is the only state that triggers downstream events (recording, alerts).
A track that goes ``TRACKER_MAX_AGE`` frames without an update is moved to
LOST; a LOST track that gets matched again returns to CONFIRMED. Tracks
that remain LOST or TENTATIVE for too long are deleted.

Detection-to-track association is performed in two phases:
  1. IoU matching - precise bounding-box overlap.
  2. Center-distance fallback - needed when FRAME_SKIP > 1 causes the
     object to "jump" by more than the IoU threshold tolerates.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np

import config
from object_detector import Detection


class TrackState(Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"


@dataclass
class Track:
    """Represents a tracked object."""
    track_id: int
    class_id: int
    class_name: str
    state: TrackState = TrackState.TENTATIVE
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    confidence_history: List[float] = field(default_factory=list)
    bbox_history: List[Tuple[int, int, int, int]] = field(default_factory=list)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    @property
    def current_bbox(self) -> Tuple[int, int, int, int]:
        return self.bbox_history[-1] if self.bbox_history else (0, 0, 0, 0)

    @property
    def current_confidence(self) -> float:
        return self.confidence_history[-1] if self.confidence_history else 0.0

    @property
    def median_confidence(self) -> float:
        if not self.confidence_history:
            return 0.0
        sorted_conf = sorted(self.confidence_history)
        n = len(sorted_conf)
        if n % 2 == 0:
            return (sorted_conf[n // 2 - 1] + sorted_conf[n // 2]) / 2
        return sorted_conf[n // 2]

    @property
    def center(self) -> Tuple[int, int]:
        x, y, w, h = self.current_bbox
        return (x + w // 2, y + h // 2)

    @property
    def duration(self) -> float:
        return self.last_seen - self.first_seen

    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED

    def to_detection(self) -> Detection:
        x, y, w, h = self.current_bbox
        return Detection(
            x=x, y=y, width=w, height=h,
            class_id=self.class_id, class_name=self.class_name,
            confidence=self.median_confidence,
        )


class ObjectTracker:
    """Tracks detected objects across frames."""

    # Maximum center-distance for the fallback matching phase, in pixels at
    # PROCESSING_WIDTH scale. Tuned empirically for FRAME_SKIP <= 4.
    MAX_CENTER_DISTANCE = 400.0

    def __init__(self):
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
        self.frame_count = 0

    def update(self, detections: List[Detection]) -> List[Track]:
        """Advance the tracker by one frame using the given detections."""
        self.frame_count += 1
        current_time = time.time()

        # Age all existing tracks
        for track in self.tracks.values():
            track.time_since_update += 1
            track.age += 1

        matched_tracks, unmatched_detections = self._match_detections(detections)

        for track_id, detection in matched_tracks:
            self._update_track(track_id, detection, current_time)

        for detection in unmatched_detections:
            self._create_track(detection, current_time)

        self._mark_lost_tracks()
        self._remove_dead_tracks()

        return [t for t in self.tracks.values() if t.is_confirmed()]

    def _match_detections(
        self,
        detections: List[Detection],
    ) -> Tuple[List, List]:
        """Associate detections with existing tracks (two-phase matching)."""
        matched = []
        unmatched = []
        used_tracks = set()

        # Process highest-confidence detections first
        sorted_detections = sorted(detections, key=lambda d: d.confidence, reverse=True)

        for detection in sorted_detections:
            best_track_id = None
            best_iou = 0.0
            best_dist = float("inf")

            # Phase 1: IoU matching
            for track_id, track in self.tracks.items():
                if track_id in used_tracks or track.state == TrackState.LOST:
                    continue
                if track.class_id != detection.class_id:
                    continue
                iou = self._compute_iou(track.current_bbox, detection.bbox)
                if iou > best_iou and iou >= config.TRACKER_IOU_THRESHOLD:
                    best_iou = iou
                    best_track_id = track_id

            # Phase 2: Center-distance fallback (handles FRAME_SKIP jumps)
            if best_track_id is None:
                for track_id, track in self.tracks.items():
                    if track_id in used_tracks or track.state == TrackState.LOST:
                        continue
                    if track.class_id != detection.class_id:
                        continue
                    dist = self._compute_center_distance(track.current_bbox, detection.bbox)
                    if dist < best_dist and dist <= self.MAX_CENTER_DISTANCE:
                        best_dist = dist
                        best_track_id = track_id

            if best_track_id is not None:
                matched.append((best_track_id, detection))
                used_tracks.add(best_track_id)
            else:
                unmatched.append(detection)

        return matched, unmatched

    def _update_track(self, track_id: int, detection: Detection, current_time: float):
        """Update an existing track with a new matching detection."""
        track = self.tracks[track_id]
        track.hits += 1
        track.time_since_update = 0
        track.last_seen = current_time
        track.confidence_history.append(detection.confidence)
        track.bbox_history.append(detection.bbox)

        # Keep history bounded to limit memory usage on long-running sessions
        max_history = config.TRACKER_MAX_HISTORY
        if len(track.confidence_history) > max_history:
            track.confidence_history = track.confidence_history[-max_history:]
            track.bbox_history = track.bbox_history[-max_history:]

        # Promote TENTATIVE -> CONFIRMED once we have enough hits
        if track.state == TrackState.TENTATIVE and track.hits >= config.TRACKER_MIN_HITS:
            track.state = TrackState.CONFIRMED

    def _create_track(self, detection: Detection, current_time: float):
        """Create a new TENTATIVE track from an unmatched detection."""
        track = Track(
            track_id=self.next_id,
            class_id=detection.class_id,
            class_name=detection.class_name,
            confidence_history=[detection.confidence],
            bbox_history=[detection.bbox],
            first_seen=current_time,
            last_seen=current_time,
        )
        self.tracks[self.next_id] = track
        self.next_id += 1

    def _mark_lost_tracks(self):
        """Move tracks that have not been updated recently to LOST state."""
        for track in self.tracks.values():
            if track.state != TrackState.LOST and track.time_since_update > config.TRACKER_MAX_AGE:
                track.state = TrackState.LOST

    def _remove_dead_tracks(self):
        """Drop tracks that have been TENTATIVE or LOST for too long."""
        to_remove = []
        for track_id, track in self.tracks.items():
            if (track.state == TrackState.TENTATIVE
                    and track.time_since_update > config.TRACKER_TENTATIVE_MAX_AGE):
                to_remove.append(track_id)
            elif (track.state == TrackState.LOST
                    and track.time_since_update > config.TRACKER_LOST_MAX_AGE):
                to_remove.append(track_id)
        for track_id in to_remove:
            del self.tracks[track_id]

    def _compute_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Intersection over Union of two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union = w1 * h1 + w2 * h2 - intersection
        return intersection / union if union > 0 else 0.0

    def _compute_center_distance(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Euclidean distance between bbox centers - fallback for FRAME_SKIP."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        cx1, cy1 = x1 + w1 / 2, y1 + h1 / 2
        cx2, cy2 = x2 + w2 / 2, y2 + h2 / 2
        return float(np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2))

    # -------------------------------------------------------------------------
    # Convenience accessors
    # -------------------------------------------------------------------------

    def get_all_tracks(self) -> List[Track]:
        return list(self.tracks.values())

    def get_confirmed_tracks(self) -> List[Track]:
        return [t for t in self.tracks.values() if t.is_confirmed()]

    def get_track_count(self) -> Dict[str, int]:
        counts = {"tentative": 0, "confirmed": 0, "lost": 0, "total": len(self.tracks)}
        for track in self.tracks.values():
            counts[track.state.value] += 1
        return counts

    def reset(self):
        self.tracks.clear()
        self.next_id = 1
        self.frame_count = 0
