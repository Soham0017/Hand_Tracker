import cv2
import mediapipe as mp
import os
import time
import numpy as np
import hand_track_module as htm

#####################
brushThickness = 15
eraserThickness = 75
#####################

folderPath = "Palette"
myList = os.listdir(folderPath)
overlayList = []
for imgpath in myList:
    image = cv2.imread(f'{folderPath}/{imgpath}')
    overlayList.append(image)

header = overlayList[0]
drawColor = (255,113,82)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon = 0.75)
xp, yp = 0,0

imgCanvas = np.zeros((720,1280,3), np.uint8)

while True:

    # 1. import image
    success, img = cap.read()
    img = cv2.flip(img,1) #flip the image

    #2.Find Hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw = False)

    if len(lmList) != 0:

        x1, y1  = lmList[8][1:] # tip of index finger
        x2, y2 = lmList[12][1:] # tip of middle finger

        #3.Check which fingers are up
        fingers = detector.fingersUp()
        #print(fingers)

        #4. Selection mode (two fingers are up)
        if fingers[1] and fingers[2]:
            xp, yp = 0,0
            print("Selection Mode")
            # Checking for click
            if y1 <125:
                if 160<x1<280:
                    header = overlayList[0]
                    drawColor = (255,113,82)
                elif 365 <x1< 485:
                    header = overlayList[1]
                    drawColor = (49,49,255)
                elif 570 <x1< 690:
                    header = overlayList[2]
                    drawColor = (87,217,126)
                elif 775 <x1< 895:
                    header = overlayList[3]
                    drawColor = (89,222,255)
                # eraser
                elif 1030 <x1< 1170:
                    header = overlayList[4]
                    drawColor = (0,0,0)
            cv2.rectangle(img, (x1,y1-25), (x2,y2+25), drawColor, cv2.FILLED)

        #5. Drawing mode (Index finger up)
        if fingers[1] and fingers[2]==False:
            cv2.circle(img, (x1,y1), 15, drawColor, cv2.FILLED)
            print("Drawing Mode")

            if xp==0 and yp==0:
                xp,yp = x1,y1 # for very initial point

            if drawColor == (0,0,0):
                cv2.line(img, (xp,yp), (x1,y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp,yp), (x1,y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp,yp), (x1,y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp,yp), (x1,y1), drawColor, brushThickness)
            
            xp,yp = x1,y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # header image
    img[0:129, 0:1280] = header
    #img = cv2.addweighted(img, 0.5,imgCanvas,0.5,0)
    cv2.imshow("Canvas", imgCanvas)
    cv2.imshow("Image", img)
    cv2.waitKey(1)