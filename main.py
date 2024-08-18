import cv2
import numpy as np
from time import sleep

width_min = 80  # Minimum width for object detection
height_min = 80  # Minimum height for object detection
offset = 6  # Offset for detection sensitivity
vertical_pos = 550  # Position of the detection line
delay = 60  # Frame delay for video playback
count = 0  # Vehicle count

detect = []  # List to store detected object centers


# Function to calculate the center of detected objects.
def page_central(x, y, w, h):  
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


cap = cv2.VideoCapture('data/video.mp4')

BGS = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret, frame1 = cap.read()
    tempo = float(1 / delay)
    sleep(tempo)
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    img_sub = BGS.apply(blur)
    dilate = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morphological_closing = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
    morphological_closing = cv2.morphologyEx(morphological_closing, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(morphological_closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame1, (25, vertical_pos), (1200, vertical_pos), (176, 130, 39), 2)

    for (i, c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        val_cont_size = (w >= width_min) and (h >= height_min)

        if not val_cont_size:
            continue

        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        detect_central = page_central(x, y, w, h)
        detect.append(detect_central)
        cv2.circle(frame1, detect_central, 4, (0, 0, 255), -1)

        for (x, y) in detect:
            if (y < (vertical_pos + offset)) and (y > (vertical_pos - offset)):
                count += 1
                cv2.line(frame1, (25, vertical_pos), (1200, vertical_pos), (0, 127, 255), 3)
                detect.remove((x, y))
                print("No. of cars detected : " + str(count))

    cv2.putText(frame1, "VEHICLE COUNT : " + str(count), (320, 70), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 4)
    cv2.imshow("Video Original", frame1)
    cv2.imshow(" Detectar ", morphological_closing)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()
