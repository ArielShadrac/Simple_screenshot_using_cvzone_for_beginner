import cv2 as cv
from cvzone.HandTrackingModule import HandDetector
import pyscreenshot as ImageGrab

cap = cv.VideoCapture(0)
detector = HandDetector(maxHands=1, detectionCon=0.8, minTrackCon=0.8)

while True:
    _, img = cap.read()
    img = cv.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False, draw=True)

    if hands:
        if detector.fingersUp(hands[0])==[0,1,1,0,0]:
            im = ImageGrab.grab()
            im.save("shot.png")
            cv.putText(img,f'SHOTTED!!!!', (10,30), cv.FONT_HERSHEY_DUPLEX, 0.8,(255,230,114),2)

    cv.imshow('Screenshot', img)
    key = cv.waitKey(1) & 0xFF
    if key == ord("q") or key == ord("Q"):
        break

cap.release()
cv.destroyAllWindows()