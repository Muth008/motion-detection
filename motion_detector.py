#!/usr/bin/env python3
"""
Motion detection module based on MOG2 background subtraction.

The detector resizes the input frame to ``PROCESSING_WIDTH`` for performance,
applies the MOG2 background subtractor, removes shadow pixels, runs
opening + closing morphology to clean up the foreground mask, extracts
contours, and filters them by area, dimensions and aspect ratio.

Bounding boxes are rescaled back to the original frame coordinates before
being returned to the caller.
"""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

import config


@dataclass
class MotionRegion:
    """Represents a single detected motion region in original-frame coordinates."""
    x: int
    y: int
    width: int
    height: int
    area: float
    contour: np.ndarray = None

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 0

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)


class MotionDetector:
    """Motion detector based on MOG2 background subtraction."""

    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.MOG2_HISTORY,
            varThreshold=config.MOG2_VAR_THRESHOLD,
            detectShadows=config.MOG2_DETECT_SHADOWS,
        )
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.MORPH_KERNEL_SIZE, config.MORPH_KERNEL_SIZE),
        )
        self.scale_ratio = 1.0

    def detect(self, frame: np.ndarray) -> Tuple[List[MotionRegion], dict]:
        """
        Run motion detection on a single frame.

        Returns:
            (regions, debug_info) where ``regions`` is the list of accepted
            motion regions (in original frame coordinates) and ``debug_info``
            contains intermediate masks for visualization.
        """
        processed, self.scale_ratio = self._preprocess(frame)
        fg_mask = self.bg_subtractor.apply(
            processed, learningRate=config.MOG2_LEARNING_RATE
        )

        if config.MOG2_DETECT_SHADOWS:
            fg_mask_no_shadows = self._remove_shadows(fg_mask)
        else:
            fg_mask_no_shadows = fg_mask

        fg_mask_clean = self._apply_morphology(fg_mask_no_shadows)
        contours, _ = cv2.findContours(
            fg_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        regions = self._filter_contours(contours)

        debug_info = {
            "raw_mask": fg_mask,
            "no_shadows_mask": fg_mask_no_shadows,
            "clean_mask": fg_mask_clean,
            "contours_count": len(contours),
            "filtered_count": len(regions),
        }

        return regions, debug_info

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """Downscale to PROCESSING_WIDTH and apply Gaussian blur."""
        height, width = frame.shape[:2]
        ratio = config.PROCESSING_WIDTH / width
        new_height = int(height * ratio)
        small = cv2.resize(frame, (config.PROCESSING_WIDTH, new_height))
        blurred = cv2.GaussianBlur(
            small, (config.BLUR_KERNEL_SIZE, config.BLUR_KERNEL_SIZE), 0
        )
        return blurred, ratio

    def _remove_shadows(self, mask: np.ndarray) -> np.ndarray:
        """Drop pixels marked as shadow by MOG2 (default value 127)."""
        result = mask.copy()
        result[result == config.MOG2_SHADOW_VALUE] = 0
        return result

    def _apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """Opening removes noise, closing fills small gaps inside objects."""
        opened = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, self.morph_kernel,
            iterations=config.MORPH_OPEN_ITERATIONS,
        )
        closed = cv2.morphologyEx(
            opened, cv2.MORPH_CLOSE, self.morph_kernel,
            iterations=config.MORPH_CLOSE_ITERATIONS,
        )
        return closed

    def _filter_contours(self, contours) -> List[MotionRegion]:
        """Drop contours that fail area, size or aspect-ratio constraints."""
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < config.MIN_CONTOUR_AREA or area > config.MAX_CONTOUR_AREA:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < config.MIN_WIDTH or h < config.MIN_HEIGHT:
                continue
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < config.MIN_ASPECT_RATIO or aspect_ratio > config.MAX_ASPECT_RATIO:
                continue

            # Rescale coordinates back to the original frame
            orig_x = int(x / self.scale_ratio)
            orig_y = int(y / self.scale_ratio)
            orig_w = int(w / self.scale_ratio)
            orig_h = int(h / self.scale_ratio)

            regions.append(
                MotionRegion(
                    x=orig_x, y=orig_y, width=orig_w, height=orig_h,
                    area=area, contour=contour,
                )
            )
        return regions

    def get_background(self) -> np.ndarray:
        """Return the current background model image (for debugging)."""
        return self.bg_subtractor.getBackgroundImage()
