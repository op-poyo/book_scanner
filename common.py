"""Helpers shared by the scripts in this repo."""
import sys

import cv2
import numpy

GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)


def open_camera(index=0, width=1280, height=720):
    """Open the webcam at the requested resolution (exits if it can't be opened)."""
    source = cv2.VideoCapture(index)
    if not source.isOpened():
        print("Error: Could not open camera source.")
        sys.exit(1)
    source.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    source.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual_w = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"camera resolution: {actual_w}x{actual_h}")
    return source


def read_key():
    """Wait 1 ms for a key press. Returns it lower-cased, 'esc' for Escape, '' if none."""
    key = cv2.waitKey(1) & 0xFF
    if key == 255:
        return ""
    if key == 27:
        return "esc"
    return chr(key).lower()


def adjust_blur(key, blur_w, blur_h):
    """w/s change the blur kernel height, d/a change its width (kernel sizes stay odd)."""
    if key == "w":
        blur_h += 2
    elif key == "s" and blur_h > 1:
        blur_h -= 2
    elif key == "d":
        blur_w += 2
    elif key == "a" and blur_w > 1:
        blur_w -= 2
    return blur_w, blur_h


def adjust_canny(key, canny_l, canny_h):
    """i/k change the high Canny threshold, j/l change the low one (low stays below high)."""
    if key == "i":
        canny_h += 1
    elif key == "k" and canny_h > canny_l:
        canny_h -= 1
    elif key == "j" and canny_l > 0:
        canny_l -= 1
    elif key == "l" and canny_l < canny_h - 1:
        canny_l += 1
    return canny_l, canny_h


def update_edge_memory(edges, memory, decay, boost):
    """Accumulate edge pixels over time: old edges fade by `decay`, new ones are boosted."""
    if memory is None:
        memory = numpy.zeros_like(edges, dtype=numpy.float32)
    memory *= decay
    edge_mask = edges > 0
    memory[edge_mask] = numpy.minimum(memory[edge_mask] + boost, 1.0)
    return memory


def order_corners(points):
    """Return four points as top-left, top-right, bottom-right, bottom-left.

    Contour approximation can start at any vertex, so without a fixed order the
    smoothing step would blend unrelated corners together between frames.
    """
    pts = numpy.asarray(points, dtype=numpy.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = numpy.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[numpy.argsort(angles)]  # clockwise on screen (image y axis points down)
    start = int(numpy.argmin(pts.sum(axis=1)))  # top-left has the smallest x + y
    return numpy.roll(pts, -start, axis=0)


def draw_quad(frame, corners, color=GREEN, mark_corners=False):
    """Draw a translucent filled quad with an outline (modifies `frame` in place)."""
    if corners is None:
        return
    polygon = corners.astype(numpy.int32).reshape(-1, 1, 2)
    overlay = frame.copy()
    cv2.drawContours(overlay, [polygon], -1, color, cv2.FILLED)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    cv2.drawContours(frame, [polygon], -1, color, 3)
    if mark_corners:
        for x, y in polygon.reshape(-1, 2):
            cv2.circle(frame, (int(x), int(y)), 6, RED, -1)


def put_status(frame, lines):
    """Write status lines in the top-left corner (with a dark outline for legibility)."""
    for i, text in enumerate(lines):
        origin = (10, 25 + 25 * i)
        cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)
