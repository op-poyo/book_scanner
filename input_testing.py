"""Exploration script: try blur / Canny / edge-memory filters on a live webcam feed.

Keys:  1-5  switch view (raw, grey, blur, canny, edge memory)
       w/s  blur kernel height +/-     d/a  blur kernel width +/-
       i/k  Canny high threshold +/-   j/l  Canny low threshold +/-
       0 or Esc  quit
"""
import cv2

from common import adjust_blur, adjust_canny, open_camera, read_key, update_edge_memory

PREVIEW = 0   # raw camera preview
GREY = 1      # greyscale
BLUR = 2      # blur filters
CANNY = 3     # Canny edge detector
CONTOUR = 4   # edge memory (edges that fade over time)

MODE_KEYS = {"1": PREVIEW, "2": GREY, "3": BLUR, "4": CANNY, "5": CONTOUR}
NEEDS_BILATERAL = (BLUR, CANNY, CONTOUR)

# Edge memory: values are fractions of full brightness per frame.
DECAY = 0.8   # must be < 1, otherwise remembered edges never fade
BOOST = 1.0
DISPLAY_THRESHOLD = 0.3


def show_blur(frame, gray, bilateral, blur_w, blur_h):
    cv2.imshow("raw blur", cv2.blur(frame, (blur_w, blur_h)))
    cv2.imshow("grey blur", cv2.blur(gray, (blur_w, blur_h)))
    cv2.imshow("bilateral", bilateral)


def show_canny(frame, gray, bilateral, blur_w, blur_h, canny_l, canny_h):
    blur = (blur_w, blur_h)
    cv2.imshow("raw canny", cv2.Canny(frame, canny_l, canny_h))
    cv2.imshow("grey canny", cv2.Canny(gray, canny_l, canny_h))
    cv2.imshow("blurry canny", cv2.Canny(cv2.blur(frame, blur), canny_l, canny_h))
    cv2.imshow("grey blurry canny", cv2.Canny(cv2.blur(gray, blur), canny_l, canny_h))
    cv2.imshow("grey bilateral canny", cv2.Canny(bilateral, canny_l, canny_h))


def show_edge_memory(edges, edge_memory):
    cv2.imshow("raw edges", edges)
    cv2.imshow("fading memory", (edge_memory * 255).astype("uint8"))
    cv2.imshow("thresholded memory", ((edge_memory >= DISPLAY_THRESHOLD) * 255).astype("uint8"))


def main():
    print(cv2.__version__)
    source = open_camera()

    mode = PREVIEW
    blur_w, blur_h = 13, 13
    canny_l, canny_h = 80, 150
    edge_memory = None

    try:
        while True:
            has_frame, frame = source.read()
            if not has_frame:
                break
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bilateral = cv2.bilateralFilter(gray, 9, 75, 75) if mode in NEEDS_BILATERAL else None

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

            key = read_key()
            if key in ("0", "esc"):
                break
            elif key in ("w", "a", "s", "d"):
                blur_w, blur_h = adjust_blur(key, blur_w, blur_h)
                print(f"w: {blur_w} h: {blur_h}")
            elif key in ("i", "j", "k", "l"):
                canny_l, canny_h = adjust_canny(key, canny_l, canny_h)
                print(f"canny_l = {canny_l}, canny_h = {canny_h}")
            elif key in MODE_KEYS:
                cv2.destroyAllWindows()
                mode = MODE_KEYS[key]
                edge_memory = None
    finally:
        source.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
