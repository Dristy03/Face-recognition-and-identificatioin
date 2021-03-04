import cv2
import numpy as np
import pickle

cascade_classifier = cv2.CascadeClassifier('src/cascade/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detections = cascade_classifier.detectMultiScale(gray, 1.5, 5)

    for (x, y, w, h) in detections:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        id_, conf = recognizer.predict(roi_gray)
        if 45 <= conf <= 85:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(img, name, (x-10, y-10), font, 1, color, stroke, cv2.LINE_AA)

        # img_item = "my-image.png"
        # cv2.imwrite(img_item, roi_gray)
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow("Result", img)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
