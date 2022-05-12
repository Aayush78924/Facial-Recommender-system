import cv2
from deepface import DeepFace
import time
from playsound import playsound


faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError('webcam cannot be open')

prevEmotion = ""
audioSelected = False
captureFrame = ""
startTime = time.time()

while True:
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()

    if not ret or audioSelected:
        continue

    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

    currTime = time.time()
    countdown = 5 - int(currTime - startTime)
    if result['dominant_emotion'] == prevEmotion and not countdown:
        try:
            path = f'songs/{prevEmotion}.mp3'
            print(path)
            playsound(path)
            audioSelected = True
        except:
            print('song not available')

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 40)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame,
                result['dominant_emotion'],
                (50, 50),
                font, 1,
                (0, 255, 0),
                2,
                cv2.LINE_4)

    if countdown > 0:
        cv2.putText(frame,
                    f'{countdown}',
                    (100, 200),
                    font, 3,
                    (0, 255, 0),
                    2,
                    cv2.LINE_4)
    cv2.imshow('Capturing', frame)
    prevEmotion = result['dominant_emotion']

cap.release()
cv2.destroyAllWindows()
