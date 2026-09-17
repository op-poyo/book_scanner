import sys
import cv2
import numpy

print(cv2.__version__)

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

# min size as fraction
MIN_AREA_FRACTION = 0.05
min_page_area = MIN_AREA_FRACTION * frame_width * frame_height

PREVIEW = 0  # Preview Mode
GREY    = 1  # Greyscale Filter
BLUR    = 2  # Blur Filter
CANNY   = 3  # Canny Edge Detector
CONTOUR = 4  # Edge Memory / Contour Mode
BOOK    = 5  # Book/Page Highlight Mode

image_filter = PREVIEW
alive = True
blur_w = 13
blur_h = 13
canny_l = 80
canny_h = 150
edge_memory = None
DECAY = 0.8  
BOOST = 1.0
DISPLAY_THRESHOLD = 0.3


smoothed_corners = None       
CORNER_SMOOTHING = 0.4       
frames_since_seen = 0
MAX_FRAMES_WITHOUT_DETECTION = 10  

# cycle which channel to use for edge detection
CHANNEL_GRAY = 0
CHANNEL_SATURATION = 1
CHANNEL_LAB_L = 2
CHANNEL_NAMES = {
    CHANNEL_GRAY: "grayscale",
    CHANNEL_SATURATION: "HSV saturation",
    CHANNEL_LAB_L: "LAB L",
}
active_channel = CHANNEL_GRAY


def get_channel(frame, gray, channel):
    # return the appropriate channel 
    if channel == CHANNEL_SATURATION:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return hsv[:, :, 1]
    elif channel == CHANNEL_LAB_L:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        return lab[:, :, 0]
    return gray


def update_edge_memory(edges, memory, decay, boost):
    if memory is None:
        memory = numpy.zeros_like(edges, dtype=numpy.float32)
    memory *= decay
    edge_mask = edges > 0
    memory[edge_mask] = numpy.minimum(memory[edge_mask] + boost, 1.0)
    return memory


def find_page_contour(edges, min_area):
    # find contours in the edge image, filter by area, and sort by area descending
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx

    return None


def smooth_contour(new_contour, previous_corners, smoothing):
    new_corners = new_contour.reshape(4, 2).astype(numpy.float32)
    if previous_corners is None or previous_corners.shape != new_corners.shape:
        return new_corners
    return smoothing * new_corners + (1 - smoothing) * previous_corners


try:
    while alive:
        has_frame, frame = source.read()
        if not has_frame:
            break

        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bilateral = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

        if image_filter == PREVIEW:
            cv2.imshow('raw', frame)

        elif image_filter == CANNY:
            rawCanny = cv2.Canny(frame, canny_l, canny_h)
            greyCanny = cv2.Canny(gray, canny_l, canny_h)
            greyBlurCanny = cv2.Canny(cv2.blur(gray, (blur_w, blur_h)), canny_l, canny_h)
            blurCanny = cv2.Canny(cv2.blur(frame, (blur_w, blur_h)), canny_l, canny_h)
            greyBilateralCanny = cv2.Canny(bilateral, canny_l, canny_h)

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab_l = lab[:, :, 0]

            saturationBilateral = cv2.bilateralFilter(saturation, d=9, sigmaColor=75, sigmaSpace=75)
            labLBilateral = cv2.bilateralFilter(lab_l, d=9, sigmaColor=75, sigmaSpace=75)
            saturationCanny = cv2.Canny(saturationBilateral, canny_l, canny_h)
            labLCanny = cv2.Canny(labLBilateral, canny_l, canny_h)

            cv2.imshow('raw canny', rawCanny)
            cv2.imshow('grey canny', greyCanny)
            cv2.imshow('blurry canny', blurCanny)
            cv2.imshow('grey blurry canny', greyBlurCanny)
            cv2.imshow('grey bilateral canny', greyBilateralCanny)
            cv2.imshow('saturation bilateral canny', saturationCanny)
            cv2.imshow('LAB L bilateral canny', labLCanny)

        elif image_filter == BLUR:
            blurry = cv2.blur(frame, (blur_w, blur_h))
            greyBlurry = cv2.blur(gray, (blur_w, blur_h))
            cv2.imshow('raw blur', blurry)
            cv2.imshow('grey blur', greyBlurry)
            cv2.imshow('bilateral', bilateral)

        elif image_filter == GREY:
            cv2.imshow('grey', gray)

        elif image_filter == CONTOUR:
            smoothed = cv2.bilateralFilter(gray, 9, 75, 75)
            edges = cv2.Canny(smoothed, canny_l, canny_h)
            edge_memory = update_edge_memory(edges, edge_memory, DECAY, BOOST)
            faded_display = (edge_memory * 255).astype(numpy.uint8)
            thresholded_display = ((edge_memory >= DISPLAY_THRESHOLD) * 255).astype(numpy.uint8)
            cv2.imshow('raw edges', edges)
            cv2.imshow('fading memory', faded_display)
            cv2.imshow('thresholded memory', thresholded_display)

        elif image_filter == BOOK:
            channel_img = get_channel(frame, gray, active_channel)
            smoothed = cv2.bilateralFilter(channel_img, 9, 75, 75)
            edges = cv2.Canny(smoothed, canny_l, canny_h)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            page_contour = find_page_contour(closed_edges, min_page_area)
            display_frame = frame.copy()

            if page_contour is not None:
                smoothed_corners = smooth_contour(page_contour, smoothed_corners, CORNER_SMOOTHING)
                frames_since_seen = 0
            else:
                frames_since_seen += 1
                if frames_since_seen > MAX_FRAMES_WITHOUT_DETECTION:
                    smoothed_corners = None

            if smoothed_corners is not None:
                draw_contour = smoothed_corners.astype(numpy.int32).reshape(-1, 1, 2)
                overlay = display_frame.copy()
                cv2.drawContours(overlay, [draw_contour], -1, (0, 255, 0), cv2.FILLED)
                cv2.addWeighted(overlay, 0.35, display_frame, 0.65, 0, display_frame)

                cv2.drawContours(display_frame, [draw_contour], -1, (0, 255, 0), 3)

            cv2.putText(display_frame, f"channel: {CHANNEL_NAMES[active_channel]} (press c to cycle)",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('book recognition', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("0") or key == 27:
            alive = False
        elif key in (ord("w"), ord("W"), ord("S"), ord("s"), ord("A"), ord("a"), ord("D"), ord("d")):
            if key == ord("W") or key == ord("w"):
                blur_h += 2
            elif key == ord("s") or key == ord("S"):
                if blur_h > 1:
                    blur_h -= 2
            elif key == ord("D") or key == ord("d"):
                blur_w += 2
            elif key == ord("A") or key == ord("a"):
                if blur_w > 1:
                    blur_w -= 2
            print(f'w: {blur_w} h: {blur_h}')
        elif key in (ord("I"), ord('i'), ord("J"), ord("j"), ord("K"), ord("k"), ord("L"), ord("l")):
            if key == ord("I") or key == ord("i"):
                canny_h += 1
            elif key == ord("K") or key == ord("k"):
                if canny_h > canny_l:
                    canny_h -= 1
            elif key == ord("J") or key == ord("j"):
                if canny_l > 0:
                    canny_l -= 1
            elif key == ord("l") or key == ord("L"):
                if canny_l < canny_h - 1:
                    canny_l += 1
            print(f"canny_l = {canny_l}, canny_h = {canny_h}")
        elif key in (ord("c"), ord("C")):
            active_channel = (active_channel + 1) % len(CHANNEL_NAMES)
            smoothed_corners = None
            frames_since_seen = 0
            edge_memory = None
            print(f"active channel: {CHANNEL_NAMES[active_channel]}")
        elif key in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5"), ord("6")):
            cv2.destroyAllWindows()
            if key == ord("6"):
                image_filter = BOOK
            elif key == ord("5"):
                image_filter = CONTOUR
            elif key == ord("4"):
                image_filter = CANNY
            elif key == ord("3"):
                image_filter = BLUR
            elif key == ord("2"):
                image_filter = GREY
            elif key == ord("1"):
                image_filter = PREVIEW
            edge_memory = None

finally:
    source.release()
    cv2.destroyAllWindows()