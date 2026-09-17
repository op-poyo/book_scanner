import sys
import cv2
import numpy
from docaligner import DocAligner

print("Loading DocAligner model...")
model = DocAligner()
print("Model loaded.")

# Initialize CLAHE contrast enhancement to prevent cold-start detection failures
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

source = cv2.VideoCapture(0)

if not source.isOpened():
    print("Error: Could not open camera source.")
    sys.exit()

base_width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
base_height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
base_aspect_ratio = round(base_width / base_height, 2)
print(f"base width: {base_width}, base height: {base_height}, base aspect ratio: {base_aspect_ratio}")

output_width = 1280
output_height = 720

source.set(cv2.CAP_PROP_FRAME_WIDTH, output_width)
source.set(cv2.CAP_PROP_FRAME_HEIGHT, output_height)

frame_width = int(source.get(3))
frame_height = int(source.get(4))

alive = True

MODE_SINGLE = 0
MODE_SPLIT = 1
MODE_NAMES = {MODE_SINGLE: "single", MODE_SPLIT: "split"}
mode = MODE_SINGLE

CORNER_SMOOTHING = 0.65
DOWNWEIGHT_SMOOTHING = 0.35
smoothing_enabled = True

MAX_FRAMES_WITHOUT_DETECTION = 20
MAX_CENTROID_JUMP_RATIO = 3.5
MIN_AREA_RATIO = 0.05
MAX_AREA_RATIO = 6.0

# Set refresh interval to 1 for continuous re-scanning every frame[cite: 2]
ROUGH_BBOX_REFRESH_INTERVAL = 1
ROUGH_BBOX_PADDING = 0.15
ROUGH_BBOX_MAX_MISSES = 10
REGION_OVERLAP_FRACTION = 0.05

FRAME_PAD = 50

tracker_state = {
    "full": {"smoothed_corners": None, "last_raw": None, "frames_since_seen": 0},
    "left": {"smoothed_corners": None, "last_raw": None, "frames_since_seen": 0},
    "right": {"smoothed_corners": None, "last_raw": None, "frames_since_seen": 0},
}

rough_bbox = None
frames_since_rough_refresh = ROUGH_BBOX_REFRESH_INTERVAL
rough_bbox_misses = 0


def detect_padded(img):
    if img is None or img.size == 0:
        return None
    
    # Apply contrast equalization to sharpen edges for the neural network
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    padded = cv2.copyMakeBorder(enhanced_bgr, FRAME_PAD, FRAME_PAD, FRAME_PAD, FRAME_PAD, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    corners = model(padded)
    if corners is not None and len(corners) == 4:
        corners = numpy.asarray(corners, dtype=numpy.float32) - FRAME_PAD
        return corners
    return None


def smooth_contour(new_corners, previous_corners, smoothing):
    new_corners = numpy.asarray(new_corners, dtype=numpy.float32)
    if previous_corners is None or previous_corners.shape != new_corners.shape:
        return new_corners
    return smoothing * new_corners + (1 - smoothing) * previous_corners


def is_plausible(new_corners, reference_corners):
    if reference_corners is None:
        return True

    new_centroid = new_corners.mean(axis=0)
    ref_centroid = reference_corners.mean(axis=0)
    ref_diag = numpy.linalg.norm(reference_corners[0] - reference_corners[2])
    if ref_diag < 1e-3:
        return True

    jump = numpy.linalg.norm(new_centroid - ref_centroid)
    if jump > MAX_CENTROID_JUMP_RATIO * ref_diag:
        return False

    new_area = cv2.contourArea(new_corners.astype(numpy.float32).reshape(-1, 1, 2))
    ref_area = cv2.contourArea(reference_corners.astype(numpy.float32).reshape(-1, 1, 2))
    if ref_area < 1e-3:
        return True

    ratio = new_area / ref_area
    return MIN_AREA_RATIO <= ratio <= MAX_AREA_RATIO


def update_tracker(state, corners):
    if corners is None or len(corners) != 4:
        state["frames_since_seen"] += 1
        if state["frames_since_seen"] > MAX_FRAMES_WITHOUT_DETECTION:
            state["smoothed_corners"] = None
            state["last_raw"] = None
        return state["smoothed_corners"]

    corners = numpy.asarray(corners, dtype=numpy.float32)
    
    # Snap immediately if target was lost or newly captured
    if state["smoothed_corners"] is None or state["frames_since_seen"] > 1:
        state["smoothed_corners"] = corners
    else:
        plausible = is_plausible(corners, state["last_raw"])
        weight = CORNER_SMOOTHING if plausible else DOWNWEIGHT_SMOOTHING

        if smoothing_enabled:
            state["smoothed_corners"] = smooth_contour(corners, state["smoothed_corners"], weight)
        else:
            state["smoothed_corners"] = corners

    state["last_raw"] = corners
    state["frames_since_seen"] = 0
    return state["smoothed_corners"]


def bounding_box(corners, shape, padding):
    x_min, y_min = corners.min(axis=0)
    x_max, y_max = corners.max(axis=0)
    pad_x = (x_max - x_min) * padding
    pad_y = (y_max - y_min) * padding
    
    x_min = max(0, int(x_min - pad_x))
    y_min = max(0, int(y_min - pad_y))
    x_max = min(shape[1], int(x_max + pad_x))
    y_max = min(shape[0], int(y_max + pad_y))
    return x_min, y_min, x_max, y_max


def refresh_rough_bbox(frame):
    global rough_bbox, frames_since_rough_refresh, rough_bbox_misses
    detected = detect_padded(frame)
    frames_since_rough_refresh = 0
    if detected is not None:
        rough_bbox = bounding_box(detected, frame.shape, ROUGH_BBOX_PADDING)
        rough_bbox_misses = 0
    else:
        rough_bbox_misses += 1
        if rough_bbox_misses > ROUGH_BBOX_MAX_MISSES:
            rough_bbox = None


def get_page_regions(frame):
    x_min, y_min, x_max, y_max = rough_bbox
    box_width = x_max - x_min
    mid_x = x_min + box_width // 2
    overlap = int(box_width * REGION_OVERLAP_FRACTION)

    left_x0, left_x1 = x_min, min(x_max, mid_x + overlap)
    right_x0, right_x1 = max(x_min, mid_x - overlap), x_max

    left_crop = frame[y_min:y_max, left_x0:left_x1]
    right_crop = frame[y_min:y_max, right_x0:right_x1]

    return (left_crop, (left_x0, y_min)), (right_crop, (right_x0, y_min))


def detect_in_region(crop, offset):
    corners = detect_padded(crop)
    if corners is None:
        return None
    corners[:, 0] += offset[0]
    corners[:, 1] += offset[1]
    return corners


def draw_quad(display_frame, corners, color):
    if corners is None:
        return
    draw_contour = corners.astype(numpy.int32).reshape(-1, 1, 2)
    overlay = display_frame.copy()
    cv2.drawContours(overlay, [draw_contour], -1, color, cv2.FILLED)
    cv2.addWeighted(overlay, 0.35, display_frame, 0.65, 0, display_frame)
    cv2.drawContours(display_frame, [draw_contour], -1, color, 3)
    for point in draw_contour.reshape(-1, 2):
        cv2.circle(display_frame, tuple(point), 6, (0, 0, 255), -1)


def reset_tracking():
    global rough_bbox, frames_since_rough_refresh, rough_bbox_misses
    for state in tracker_state.values():
        state["smoothed_corners"] = None
        state["last_raw"] = None
        state["frames_since_seen"] = 0
    rough_bbox = None
    frames_since_rough_refresh = ROUGH_BBOX_REFRESH_INTERVAL
    rough_bbox_misses = 0


try:
    while alive:
        has_frame, frame = source.read()
        if not has_frame:
            break

        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()

        if frames_since_rough_refresh >= ROUGH_BBOX_REFRESH_INTERVAL or rough_bbox is None:
            refresh_rough_bbox(frame)
        else:
            frames_since_rough_refresh += 1

        if mode == MODE_SINGLE:
            full_raw = detect_padded(frame)
            corners = update_tracker(tracker_state["full"], full_raw)
            draw_quad(display_frame, corners, (0, 255, 0))

        elif mode == MODE_SPLIT:
            if rough_bbox is None:
                # Direct full-frame split fallback when rough_bbox is uninitialized[cite: 2]
                h, w = frame.shape[:2]
                mid = w // 2
                left_raw = detect_in_region(frame[:, :mid], (0, 0))
                right_raw = detect_in_region(frame[:, mid:], (mid, 0))
                
                left_corners = update_tracker(tracker_state["left"], left_raw)
                right_corners = update_tracker(tracker_state["right"], right_raw)
            else:
                (left_crop, left_offset), (right_crop, right_offset) = get_page_regions(frame)
                left_raw = detect_in_region(left_crop, left_offset)
                right_raw = detect_in_region(right_crop, right_offset)
                left_corners = update_tracker(tracker_state["left"], left_raw)
                right_corners = update_tracker(tracker_state["right"], right_raw)

            draw_quad(display_frame, left_corners, (0, 255, 0))
            draw_quad(display_frame, right_corners, (255, 0, 0))

        if rough_bbox is not None:
            x_min, y_min, x_max, y_max = rough_bbox
            cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)

        status_mode = f"mode: {MODE_NAMES[mode]} (press m to toggle)"
        status_smooth = "smoothing: ON (press s to toggle)" if smoothing_enabled else "smoothing: OFF (press s to toggle)"
        cv2.putText(display_frame, status_mode, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_frame, status_smooth, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow('doc aligner', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("0") or key == 27:
            alive = False
        elif key in (ord("s"), ord("S")):
            smoothing_enabled = not smoothing_enabled
            reset_tracking()
            print(f"smoothing: {'ON' if smoothing_enabled else 'OFF'}")
        elif key in (ord("m"), ord("M")):
            mode = MODE_SPLIT if mode == MODE_SINGLE else MODE_SINGLE
            reset_tracking()
            print(f"mode: {MODE_NAMES[mode]}")

finally:
    source.release()
    cv2.destroyAllWindows()