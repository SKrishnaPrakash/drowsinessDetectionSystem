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
model = load_model('cnnCat2.keras')

# Start webcam
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score = 0
thicc = 2
path = os.getcwd()

# CLAHE for better contrast normalization
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

def preprocess_eye(eye_img):
    eye_img = cv2.resize(eye_img, (48, 48))
    eye_img = clahe.apply(eye_img)
    eye_img = eye_img / 255.0
    return eye_img.reshape(1, 48, 48, 1)

def predict_eye_state(eye_img):
    pred = model.predict(eye_img)
    if pred.shape[-1] == 1:  # Binary sigmoid
        return ('Closed' if pred[0][0] < 0.3 else 'Open', pred[0][0])
    else:  # Softmax
        label = np.argmax(pred)
        conf = np.max(pred)
        return ('Closed' if label == 0 and conf > 0.7 else 'Open', conf)

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height - 50), (350, height), (0, 0, 0), thickness=cv2.FILLED)

    rpred, lpred = 'Closed', 'Closed'
    rconf, lconf = 0.0, 0.0

    # Right eye: choose best candidate
    if len(right_eye) > 0:
        best_rconf = -1
        for (x, y, w, h) in right_eye:
            r_eye = gray[y:y + h, x:x + w]
            eye_input = preprocess_eye(r_eye)
            label, conf = predict_eye_state(eye_input)
            if conf > best_rconf:
                rpred, rconf = label, conf
                best_rconf = conf
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
                cv2.imshow("Right Eye", r_eye)

    # Left eye: choose best candidate
    if len(left_eye) > 0:
        best_lconf = -1
        for (x, y, w, h) in left_eye:
            l_eye = gray[y:y + h, x:x + w]
            eye_input = preprocess_eye(l_eye)
            label, conf = predict_eye_state(eye_input)
            if conf > best_lconf:
                lpred, lconf = label, conf
                best_lconf = conf
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.imshow("Left Eye", l_eye)

    # Drowsiness logic
    if rpred == 'Closed' and lpred == 'Closed':
        score += 1
        cv2.putText(frame, f"Closed ({rconf:.2f}/{lconf:.2f})", (20, height - 20), font, 1, (255, 255, 255), 1)
    else:
        score -= 1
        cv2.putText(frame, f"Open ({rconf:.2f}/{lconf:.2f})", (20, height - 20), font, 1, (255, 255, 255), 1)

    score = max(score, 0)
    cv2.putText(frame, 'Score:' + str(score), (250, height - 20), font, 1, (255, 255, 255), 1)

    if score > 15:
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        if not mixer.get_busy():
            sound.play(-1)
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