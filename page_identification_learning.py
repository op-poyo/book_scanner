"""Exploration script: classical page detection on a live webcam feed.

Same views and keys as input_testing.py, plus:
       6  page highlight (edges -> contour -> largest convex quad)
       c  cycle the input channel (grayscale / HSV saturation / LAB lightness)
"""
import cv2

from common import (adjust_blur, adjust_canny, draw_quad, open_camera, order_corners,
                    put_status, read_key, update_edge_memory)
from tracking import CornerTracker

PREVIEW = 0   # raw camera preview
GREY = 1      # greyscale
BLUR = 2      # blur filters
CANNY = 3     # Canny edge detector (several variants side by side)
CONTOUR = 4   # edge memory (edges that fade over time)
BOOK = 5      # book/page highlight

MODE_KEYS = {"1": PREVIEW, "2": GREY, "3": BLUR, "4": CANNY, "5": CONTOUR, "6": BOOK}

# Edge memory
DECAY = 0.8
BOOST = 1.0
DISPLAY_THRESHOLD = 0.3

# Page detection
MIN_AREA_FRACTION = 0.05  # minimum page size as a fraction of the frame
CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# Which channel to run edge detection on (cycled with 'c')
CHANNEL_NAMES = ["grayscale", "HSV saturation", "LAB L"]
CHANNEL_GRAY, CHANNEL_SATURATION, CHANNEL_LAB_L = range(3)


def get_channel(frame, gray, channel):
    """Return the single-channel image used for edge detection."""
    if channel == CHANNEL_SATURATION:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1]
    if channel == CHANNEL_LAB_L:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 0]
    return gray


def find_page_contour(edges, min_area):
    """Largest convex four-sided contour in an edge image, as ordered corners (or None)."""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return order_corners(approx.reshape(4, 2))
    return None


def show_blur(frame, gray, bilateral, blur_w, blur_h):
    cv2.imshow("raw blur", cv2.blur(frame, (blur_w, blur_h)))
    cv2.imshow("grey blur", cv2.blur(gray, (blur_w, blur_h)))
    cv2.imshow("bilateral", bilateral)


def show_canny(frame, gray, bilateral, blur_w, blur_h, canny_l, canny_h):
    blur = (blur_w, blur_h)
    saturation = get_channel(frame, gray, CHANNEL_SATURATION)
    lab_l = get_channel(frame, gray, CHANNEL_LAB_L)
    cv2.imshow("raw canny", cv2.Canny(frame, canny_l, canny_h))
    cv2.imshow("grey canny", cv2.Canny(gray, canny_l, canny_h))
    cv2.imshow("blurry canny", cv2.Canny(cv2.blur(frame, blur), canny_l, canny_h))
    cv2.imshow("grey blurry canny", cv2.Canny(cv2.blur(gray, blur), canny_l, canny_h))
    cv2.imshow("grey bilateral canny", cv2.Canny(bilateral, canny_l, canny_h))
    cv2.imshow("saturation bilateral canny",
               cv2.Canny(cv2.bilateralFilter(saturation, 9, 75, 75), canny_l, canny_h))
    cv2.imshow("LAB L bilateral canny",
               cv2.Canny(cv2.bilateralFilter(lab_l, 9, 75, 75), canny_l, canny_h))


def show_edge_memory(edges, edge_memory):
    cv2.imshow("raw edges", edges)
    cv2.imshow("fading memory", (edge_memory * 255).astype("uint8"))
    cv2.imshow("thresholded memory", ((edge_memory >= DISPLAY_THRESHOLD) * 255).astype("uint8"))


def show_page_highlight(frame, gray, channel, canny_l, canny_h, tracker):
    channel_img = get_channel(frame, gray, channel)
    smoothed = cv2.bilateralFilter(channel_img, 9, 75, 75)
    edges = cv2.Canny(smoothed, canny_l, canny_h)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, CLOSE_KERNEL)

    min_area = MIN_AREA_FRACTION * frame.shape[0] * frame.shape[1]
    corners = tracker.update(find_page_contour(closed_edges, min_area))

    display = frame.copy()
    draw_quad(display, corners)
    put_status(display, [f"channel: {CHANNEL_NAMES[channel]} (press c to cycle)"])
    cv2.imshow("book recognition", display)


def main():
    print(cv2.__version__)
    source = open_camera()

    mode = PREVIEW
    blur_w, blur_h = 13, 13
    canny_l, canny_h = 80, 150
    channel = CHANNEL_GRAY
    edge_memory = None
    tracker = CornerTracker(smoothing=0.4, max_missed_frames=10)

    try:
        while True:
            has_frame, frame = source.read()
            if not has_frame:
                break
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bilateral = cv2.bilateralFilter(gray, 9, 75, 75) if mode in (BLUR, CANNY, CONTOUR) else None

            if mode == PREVIEW:
                cv2.imshow("raw", frame)
            elif mode == GREY:
                cv2.imshow("grey", gray)
            elif mode == BLUR:
                show_blur(frame, gray, bilateral, blur_w, blur_h)
            elif mode == CANNY:
                show_canny(frame, gray, bilateral, blur_w, blur_h, canny_l, canny_h)
            elif mode == CONTOUR:
                edges = cv2.Canny(bilateral, canny_l, canny_h)
                edge_memory = update_edge_memory(edges, edge_memory, DECAY, BOOST)
                show_edge_memory(edges, edge_memory)
            elif mode == BOOK:
                show_page_highlight(frame, gray, channel, canny_l, canny_h, tracker)

            key = read_key()
            if key in ("0", "esc"):
                break
            elif key in ("w", "a", "s", "d"):
                blur_w, blur_h = adjust_blur(key, blur_w, blur_h)
                print(f"w: {blur_w} h: {blur_h}")
            elif key in ("i", "j", "k", "l"):
                canny_l, canny_h = adjust_canny(key, canny_l, canny_h)
                print(f"canny_l = {canny_l}, canny_h = {canny_h}")
            elif key == "c":
                channel = (channel + 1) % len(CHANNEL_NAMES)
                tracker.reset()
                edge_memory = None
                print(f"active channel: {CHANNEL_NAMES[channel]}")
            elif key in MODE_KEYS:
                cv2.destroyAllWindows()
                mode = MODE_KEYS[key]
                edge_memory = None
    finally:
        source.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
