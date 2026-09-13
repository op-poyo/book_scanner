import sys
import cv2

print(cv2.__version__)

source = cv2.VideoCapture(0)
# ret, frame = source.read()

frame_width = int(source.get(3))
frame_height = int(source.get(4))

win_name = 'Camera Preview'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

while cv2.waitKey(1) != 27: # Escape
    has_frame, frame = source.read()
    if not has_frame:
        break
    cv2.imshow(win_name, frame)

source.release()
cv2.destroyWindow(win_name)
