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


def warp_page(frame, corners):
    """Flatten the quad `corners` (TL, TR, BR, BL) into an upright rectangle.

    The output is mirrored back, because the live feed is flipped for display
    and a book viewed in a mirror would otherwise read backwards.
    Returns None if the quad is degenerate.
    """
    if corners is None:
        return None
    src = numpy.asarray(corners, dtype=numpy.float32).reshape(4, 2)
    tl, tr, br, bl = src
    width = int(round(max(numpy.linalg.norm(tr - tl), numpy.linalg.norm(br - bl))))
    height = int(round(max(numpy.linalg.norm(bl - tl), numpy.linalg.norm(br - tr))))
    if width < 8 or height < 8:
        return None
    dst = numpy.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                      dtype=numpy.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    page = cv2.warpPerspective(frame, matrix, (width, height))
    return cv2.flip(page, 1)


def fit_in_slot(image, slot_w, slot_h, label=""):
    """Letterbox `image` into a black slot_w x slot_h tile (placeholder text if None)."""
    tile = numpy.full((slot_h, slot_w, 3), 40, dtype=numpy.uint8)  # dark grey so the panel is visible
    if image is None:
        cv2.putText(tile, "no page", (10, slot_h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (120, 120, 120), 2)
    else:
        scale = min(slot_w / image.shape[1], slot_h / image.shape[0])
        new_w = max(1, int(image.shape[1] * scale))
        new_h = max(1, int(image.shape[0] * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        x0, y0 = (slot_w - new_w) // 2, (slot_h - new_h) // 2
        tile[y0:y0 + new_h, x0:x0 + new_w] = resized
    if label:
        cv2.putText(tile, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(tile, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)
    return tile


def render_pages_panel(frame, page_corners, panel_w, panel_h):
    """Tile the flattened pages side by side into one panel_w x panel_h image.

    `page_corners` is a list of (label, corners_or_None), one tile each.
    """
    slot_w = panel_w // len(page_corners)
    tiles = [fit_in_slot(warp_page(frame, corners), slot_w, panel_h, label)
             for label, corners in page_corners]
    panel = numpy.hstack(tiles)
    if panel.shape[1] < panel_w:  # integer division left a few spare pixels
        panel = numpy.pad(panel, ((0, 0), (0, panel_w - panel.shape[1]), (0, 0)))
    return panel
