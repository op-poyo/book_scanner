"""Smoothing and bookkeeping for detected page corners."""
import numpy

import cv2

# Corner smoothing: weight given to a new detection when blending it into the estimate.
CORNER_SMOOTHING = 0.65
DOWNWEIGHT_SMOOTHING = 0.35  # used instead when the detection looks like a spurious jump
MAX_FRAMES_WITHOUT_DETECTION = 20

# A detection is "implausible" if its centroid jumps by more than this many page
# diagonals, or its area changes by a factor outside [MIN_AREA_RATIO, MAX_AREA_RATIO].
MAX_CENTROID_JUMP_RATIO = 3.5
MIN_AREA_RATIO = 0.05
MAX_AREA_RATIO = 6.0

# Rough box around the page (used to crop left/right pages in split mode).
ROUGH_BBOX_PADDING = 0.15
ROUGH_BBOX_MAX_MISSES = 10


def _area(corners):
    return cv2.contourArea(corners.astype(numpy.float32).reshape(-1, 1, 2))


def is_plausible(new_corners, reference_corners):
    """Does `new_corners` look like the same page as `reference_corners`?"""
    if reference_corners is None:
        return True

    # Diagonal of the reference's bounding box (independent of corner order).
    ref_diag = numpy.linalg.norm(reference_corners.max(axis=0) - reference_corners.min(axis=0))
    if ref_diag < 1e-3:
        return True
    jump = numpy.linalg.norm(new_corners.mean(axis=0) - reference_corners.mean(axis=0))
    if jump > MAX_CENTROID_JUMP_RATIO * ref_diag:
        return False

    ref_area = _area(reference_corners)
    if ref_area < 1e-3:
        return True
    return MIN_AREA_RATIO <= _area(new_corners) / ref_area <= MAX_AREA_RATIO


class CornerTracker:
    """Exponentially smooths page corners and bridges short detection dropouts.

    - Plausible detections are blended in with weight `smoothing`; implausible ones
      (big jump or size change) only with `downweight`.
    - If the page was lost for more than one frame (or never seen) the estimate snaps
      to the new detection instead of sliding across the screen.
    - Missed frames keep returning the last estimate until `max_missed_frames` is
      exceeded, which hides single-frame dropouts.
    """

    def __init__(self, smoothing=CORNER_SMOOTHING, downweight=DOWNWEIGHT_SMOOTHING,
                 max_missed_frames=MAX_FRAMES_WITHOUT_DETECTION):
        self.smoothing = smoothing
        self.downweight = downweight
        self.max_missed_frames = max_missed_frames
        self.smoothing_enabled = True
        self.reset()

    def reset(self):
        self.smoothed = None
        self.last_raw = None
        self.frames_since_seen = 0

    def update(self, corners):
        """Feed in this frame's detection (or None); returns the current estimate."""
        if corners is None or len(corners) != 4:
            self.frames_since_seen += 1
            if self.frames_since_seen > self.max_missed_frames:
                self.reset()
            return self.smoothed

        corners = numpy.asarray(corners, dtype=numpy.float32)
        if self.smoothed is None or self.frames_since_seen > 1:
            self.smoothed = corners
        elif self.smoothing_enabled:
            plausible = is_plausible(corners, self.last_raw)
            weight = self.smoothing if plausible else self.downweight
            self.smoothed = weight * corners + (1 - weight) * self.smoothed
        else:
            self.smoothed = corners

        self.last_raw = corners
        self.frames_since_seen = 0
        return self.smoothed


def bounding_box(corners, frame_shape, padding):
    """Axis-aligned box around `corners`, grown by `padding` and clipped to the frame."""
    x_min, y_min = corners.min(axis=0)
    x_max, y_max = corners.max(axis=0)
    pad_x = (x_max - x_min) * padding
    pad_y = (y_max - y_min) * padding
    return (
        max(0, int(x_min - pad_x)),
        max(0, int(y_min - pad_y)),
        min(frame_shape[1], int(x_max + pad_x)),
        min(frame_shape[0], int(y_max + pad_y)),
    )


class RoughBox:
    """Padded bounding box (x_min, y_min, x_max, y_max) around the latest full-frame detection."""

    def __init__(self, padding=ROUGH_BBOX_PADDING, max_misses=ROUGH_BBOX_MAX_MISSES):
        self.padding = padding
        self.max_misses = max_misses
        self.reset()

    def reset(self):
        self.box = None
        self.misses = 0

    def update(self, corners, frame_shape):
        if corners is not None:
            self.box = bounding_box(corners, frame_shape, self.padding)
            self.misses = 0
            return
        self.misses += 1
        if self.misses > self.max_misses:
            self.box = None
