import numpy as np
import cv2 as cv
import dlib
import itertools
from gray_to_color import grayToColor

# Load the detector
detector = dlib.get_frontal_face_detector()

leftEye = list(range(36, 42))
rightEye = list(range(42, 48))
mouth = list(range(48, 68))

eyesAndMouth = list(itertools.chain(leftEye, rightEye, mouth))


# Load the predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# face cocordinates
x1 = 0
y1 = 0
x2 = 0
y2 = 0

# safety margin that makes face vivible
safty_margin = 30

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    color = cv.cvtColor(frame, 1)

    gausian = cv.adaptiveThreshold(
        gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2
    )

    height, width = gray.shape

    # Use detector to find landmarks
    faces = detector(gray)

    for face in faces:
        x1 = face.left()  # left point
        y1 = face.top()  # top point
        x2 = face.right()  # right point
        y2 = face.bottom()  # bottom point

        # final crop values - some margin is added in order to have
        if y1 - safty_margin * 2 < 0:
            y1f = 0
        else:
            y1f = y1 - safty_margin * 2

        if y2 + safty_margin > width:
            y2f = width
        else:
            y2f = y2 + safty_margin

        if x1 - safty_margin < 0:
            x1f = 0
        else:
            x1f = x1 - safty_margin

        if x2 + safty_margin > height:
            x2f = height
        else:
            x2f = x2 + safty_margin

        # Create landmark object
        landmarks = predictor(image=gray, box=face)

        # crop image to face
        croppedFace = gausian[y1f:y2f, x1f:x2f]
        # szkieletyzacja
        kernel = np.ones((2, 2), np.uint8)
        skeletization = cv.dilate(croppedFace, kernel, iterations=1)
        color[y1f:y2f, x1f:x2f] = grayToColor(skeletization)
        # Loop through all the points
        for n in eyesAndMouth:
            x = landmarks.part(n).x
            y = landmarks.part(n).y

            # Draw a circle
            cv.circle(
                img=color, center=(x, y), radius=2, color=(255, 0, 0), thickness=-1
            )

    # Display the resulting frame
    cv.imshow("frame", color)

    if cv.waitKey(1) == ord("q"):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
