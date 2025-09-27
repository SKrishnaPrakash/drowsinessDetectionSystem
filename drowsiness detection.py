import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

# Initialize alarm sound
mixer.init()
sound = mixer.Sound('alarm.wav')

# Load Haar cascades
face = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

# Load trained model
model = load_model('cnnCat2.keras', safe_mode=False)

# Start webcam
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score = 0
thicc = 2
path = os.getcwd()

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    rpred = 'Unknown'
    lpred = 'Unknown'

    for (x, y, w, h) in right_eye:
        r_eye = gray[y:y + h, x:x + w]
        r_eye = cv2.resize(r_eye, (48, 48))  # match training size
        r_eye = cv2.equalizeHist(r_eye)
        r_eye = r_eye / 255.0
        r_eye = r_eye.reshape(1, 48, 48, 1)
        pred = model.predict(r_eye)[0][0]
        rpred = 'Closed' if pred < 0.5 else 'Open'
        break

    for (x, y, w, h) in left_eye:
        l_eye = gray[y:y + h, x:x + w]
        l_eye = cv2.resize(l_eye, (48, 48))
        l_eye = cv2.equalizeHist(l_eye)
        l_eye = l_eye / 255.0
        l_eye = l_eye.reshape(1, 48, 48, 1)
        pred = model.predict(l_eye)[0][0]
        lpred = 'Closed' if pred < 0.5 else 'Open'
        break

    # Drowsiness logic
    if rpred == 'Closed' and lpred == 'Closed':
        score += 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        score -= 1
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    score = max(score, 0)
    cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score > 15:
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        if not mixer.get_busy():
            sound.play(-1)  # loop alarm
        thicc = thicc + 2 if thicc < 16 else thicc - 2
        thicc = max(thicc, 2)
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
    else:
        sound.stop()

    cv2.imshow('Drowsiness Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()