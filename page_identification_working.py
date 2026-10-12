"""Live page detection with DocAligner (a pretrained neural network) plus corner tracking.

Keys:  m  toggle single / split mode (split tracks the left and right page separately)
       s  toggle corner smoothing
       0, q or Esc  quit
"""
import cv2
import numpy
from docaligner import DocAligner

from common import BLUE, GREEN, draw_quad, open_camera, order_corners, put_status, read_key
from tracking import CornerTracker, RoughBox

MODE_SINGLE = "single"
MODE_SPLIT = "split"

# Detection
FRAME_PAD = 50                    # black border added so corners on the frame edge can be found
REGION_OVERLAP_FRACTION = 0.05    # left/right crops overlap by this fraction of the page width
ROUGH_BBOX_REFRESH_INTERVAL = 1   # split mode: full-frame detection every N frames (1 = every frame)


class PageDetector:
    """DocAligner wrapped with the preprocessing that makes it more reliable."""

    def __init__(self):
        print("Loading DocAligner model...")
        self.model = DocAligner()
        print("Model loaded.")
        # CLAHE contrast enhancement helps detection when the page first appears.
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _enhance(self, image):
        lightness, a, b = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
        merged = cv2.merge((self.clahe.apply(lightness), a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def detect(self, image):
        """Return the page's four corners in `image` coordinates, or None."""
        if image is None or image.size == 0:
            return None
        padded = cv2.copyMakeBorder(self._enhance(image), FRAME_PAD, FRAME_PAD, FRAME_PAD,
                                    FRAME_PAD, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        corners = self.model(padded)
        if corners is None or len(corners) != 4:
            return None
        return order_corners(numpy.asarray(corners, dtype=numpy.float32) - FRAME_PAD)


def get_page_regions(frame, box):
    """Left and right crops as (crop, (x_offset, y_offset)).

    Uses the rough box around the page, or the whole frame if there is none yet.
    """
    if box is None:
        mid = frame.shape[1] // 2
        return [(frame[:, :mid], (0, 0)), (frame[:, mid:], (mid, 0))]

    x_min, y_min, x_max, y_max = box
    width = x_max - x_min
    mid_x = x_min + width // 2
    overlap = int(width * REGION_OVERLAP_FRACTION)
    left_x0, left_x1 = x_min, min(x_max, mid_x + overlap)
    right_x0, right_x1 = max(x_min, mid_x - overlap), x_max
    return [(frame[y_min:y_max, left_x0:left_x1], (left_x0, y_min)),
            (frame[y_min:y_max, right_x0:right_x1], (right_x0, y_min))]


def detect_in_region(detector, crop, offset):
    """Detect within a crop and convert the corners back to full-frame coordinates."""
    corners = detector.detect(crop)
    if corners is None:
        return None
    return corners + numpy.asarray(offset, dtype=numpy.float32)


def main():
    detector = PageDetector()
    source = open_camera()

    mode = MODE_SINGLE
    smoothing_enabled = True
    trackers = {name: CornerTracker() for name in ("full", "left", "right")}
    rough_box = RoughBox()
    frames_since_refresh = ROUGH_BBOX_REFRESH_INTERVAL

    def reset_tracking():
        nonlocal frames_since_refresh
        for tracker in trackers.values():
            tracker.reset()
            tracker.smoothing_enabled = smoothing_enabled
        rough_box.reset()
        frames_since_refresh = ROUGH_BBOX_REFRESH_INTERVAL

    try:
        while True:
            has_frame, frame = source.read()
            if not has_frame:
                break
            frame = cv2.flip(frame, 1)
            display = frame.copy()

            # One full-frame detection per frame at most. Single mode tracks its result
            # directly; split mode uses it to place the rough box that gets cropped.
            frames_since_refresh += 1
            full_corners = None
            if (mode == MODE_SINGLE or rough_box.box is None
                    or frames_since_refresh >= ROUGH_BBOX_REFRESH_INTERVAL):
                full_corners = detector.detect(frame)
                rough_box.update(full_corners, frame.shape)
                frames_since_refresh = 0

            if mode == MODE_SINGLE:
                draw_quad(display, trackers["full"].update(full_corners), GREEN, mark_corners=True)
            else:
                (left_crop, left_offset), (right_crop, right_offset) = get_page_regions(
                    frame, rough_box.box)
                left_raw = detect_in_region(detector, left_crop, left_offset)
                right_raw = detect_in_region(detector, right_crop, right_offset)
                draw_quad(display, trackers["left"].update(left_raw), GREEN, mark_corners=True)
                draw_quad(display, trackers["right"].update(right_raw), BLUE, mark_corners=True)

            if rough_box.box is not None:
                x_min, y_min, x_max, y_max = rough_box.box
                cv2.rectangle(display, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)

            put_status(display, [f"mode: {mode} (press m to toggle)",
                                 f"smoothing: {'ON' if smoothing_enabled else 'OFF'} (press s to toggle)"])
            cv2.imshow("doc aligner", display)

            key = read_key()
            if key in ("0", "q", "esc"):
                break
            elif key == "s":
                smoothing_enabled = not smoothing_enabled
                reset_tracking()
                print(f"smoothing: {'ON' if smoothing_enabled else 'OFF'}")
            elif key == "m":
                mode = MODE_SPLIT if mode == MODE_SINGLE else MODE_SINGLE
                reset_tracking()
                print(f"mode: {mode}")
    finally:
        source.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
