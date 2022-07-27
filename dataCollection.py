import os
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time



cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

offset = 20
imgSize = 300
#此处放标签路径
folder = "E:/HandSignDetection/test/"
counter = 0
#t = 0
n=0
labels = ["0","1","2","3","4","5","6","7","8","9","A", "B", "C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
for i in range(len(labels)):
    if  not os.path.exists(os.path.join(folder,labels[i])):
        os.makedirs(os.path.join(folder,labels[i]))
while True:
    if(len(os.listdir(os.path.join(folder,str(labels[n]))))) ==300:
        n+=1
    if (len(os.listdir(os.path.join(folder,str(labels[n]))))) < 300:
        counter = len(os.listdir(os.path.join(folder,str(labels[n]))))
        t=n
        break

while True:
    success, Oimg = cap.read()
    hands, img = detector.findHands(Oimg)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        x1 = np.maximum(x-offset,0)
        x2 = np.minimum(x+w+offset,Oimg.shape[1])
        y1 = np.maximum(y-offset,0)
        y2 = np.minimum(y+h+offset,Oimg.shape[0])
        imgCrop = Oimg[y1:y2,x1:x2]
        #imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    key = cv2.waitKey(1)
    if key == ord("s"):
        if t>35:
            print("all datasets is recorded")
            break
        lable_folder = os.path.join(folder,str(labels[t]))
        counter += 1
        cv2.imwrite(f'{lable_folder}/Image_{time.time()}.jpg',imgWhite)
        print(str(labels[t]),counter)
        if counter == 300:
            t+=1
            counter=0
        

    
    cv2.imshow("Image", img)


    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()