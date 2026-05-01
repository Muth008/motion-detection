#!/usr/bin/env python3
"""
Object detection and classification using a YOLO model.

Wraps the Ultralytics YOLO API and provides two operating modes:

1. ``detect(frame)``                   - run YOLO on the whole frame (or a
                                         given region of interest).
2. ``detect_in_motion_regions(frame,   - run YOLO on each motion region
   motion_regions)``                     produced by the MOG2 stage. The
                                         results are merged via NMS to remove
                                         duplicates from overlapping regions.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

import config


@dataclass
class Detection:
    """Represents a single detected object in original-frame coordinates."""
    x: int
    y: int
    width: int
    height: int
    class_id: int
    class_name: str
    confidence: float

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        return self.width * self.height

    def __str__(self) -> str:
        return f"{self.class_name} ({self.confidence:.0%})"


class ObjectDetector:
    """YOLO-based object detector using the Ultralytics framework."""

    def __init__(self, model_path: str = None):
        if not YOLO_AVAILABLE:
            raise RuntimeError(
                "Ultralytics YOLO is not installed. Run: pip install ultralytics"
            )

        model_path = model_path or config.YOLO_MODEL
        print(f"   Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        self.relevant_classes = set(config.YOLO_RELEVANT_CLASSES)
        print(
            f"   Model loaded. Relevant classes: "
            f"{[self.class_names[i] for i in self.relevant_classes]}"
        )

    def detect(
        self,
        frame: np.ndarray,
        region: Tuple[int, int, int, int] = None,
    ) -> List[Detection]:
        """
        Run YOLO inference on a frame or a region of interest.

        Args:
            frame:  Full frame (BGR ndarray).
            region: Optional (x, y, w, h) region inside ``frame``. When given,
                    inference is run only on the cropped region and the
                    returned coordinates are translated back to the full frame.
        """
        if region is not None:
            x, y, w, h = region
            roi = frame[y:y + h, x:x + w]
            offset_x, offset_y = x, y
        else:
            roi = frame
            offset_x, offset_y = 0, 0

        imgsz = getattr(config, "YOLO_IMGSZ", 640)
        results = self.model(
            roi,
            conf=config.YOLO_CONFIDENCE_THRESHOLD,
            iou=config.YOLO_IOU_THRESHOLD,
            classes=list(self.relevant_classes) if config.YOLO_FILTER_CLASSES else None,
            imgsz=imgsz,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())

                if config.YOLO_FILTER_CLASSES and class_id not in self.relevant_classes:
                    continue

                x1, y1, x2, y2 = xyxy
                detections.append(
                    Detection(
                        x=int(x1) + offset_x,
                        y=int(y1) + offset_y,
                        width=int(x2 - x1),
                        height=int(y2 - y1),
                        class_id=class_id,
                        class_name=self.class_names[class_id],
                        confidence=conf,
                    )
                )

        return detections

    def detect_in_motion_regions(
        self,
        frame: np.ndarray,
        motion_regions: List,
    ) -> List[Detection]:
        """
        Run YOLO on each motion region (with padding) and merge results
        using non-maximum suppression to remove duplicates between overlapping
        regions.
        """
        all_detections = []
        for region in motion_regions:
            padding = config.YOLO_REGION_PADDING
            x = max(0, region.x - padding)
            y = max(0, region.y - padding)
            w = min(frame.shape[1] - x, region.width + 2 * padding)
            h = min(frame.shape[0] - y, region.height + 2 * padding)
            detections = self.detect(frame, region=(x, y, w, h))
            all_detections.extend(detections)
        return self._remove_duplicates(all_detections)

    def _remove_duplicates(
        self,
        detections: List[Detection],
        iou_threshold: float = 0.5,
    ) -> List[Detection]:
        """Greedy NMS within the same class."""
        if len(detections) <= 1:
            return detections
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        keep = []
        while sorted_dets:
            best = sorted_dets.pop(0)
            keep.append(best)
            sorted_dets = [
                d for d in sorted_dets
                if self._compute_iou(best, d) < iou_threshold or d.class_id != best.class_id
            ]
        return keep

    def _compute_iou(self, det1: Detection, det2: Detection) -> float:
        x1 = max(det1.x, det2.x)
        y1 = max(det1.y, det2.y)
        x2 = min(det1.x + det1.width, det2.x + det2.width)
        y2 = min(det1.y + det1.height, det2.y + det2.height)
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        union = det1.area + det2.area - intersection
        return intersection / union if union > 0 else 0
