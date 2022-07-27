import os
import json
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model import swin_tiny_patch4_window7_224 as create_model

cap = cv2.VideoCapture(0)
offset = 20
imgSize = 300
detector = HandDetector(maxHands=1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

img_size = 224
data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

#此处改标签集路径
json_path = 'E:/ActionDetectionforSignLanguage-main/pytorch_classification/swin_transformer/class_indices.json'
assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

with open(json_path, "r") as f:
        class_indict = json.load(f)
#num_classes标签数量
model = create_model(num_classes=6).to(device)
#模型路径
model_weight_path = "E:/ActionDetectionforSignLanguage-main/best_model.pth"
model.load_state_dict(torch.load(model_weight_path, map_location=device))
model.eval()


while True:
    success, Oimg = cap.read()
    imgOutput = Oimg.copy()
    hands, img = detector.findHands(Oimg)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        #imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        #手部图像边框出界检测
        x1 = np.maximum(x-offset,0)
        x2 = np.minimum(x+w+offset,Oimg.shape[1])
        y1 = np.maximum(y-offset,0)
        y2 = np.minimum(y+h+offset,Oimg.shape[0])
        imgCrop = img[y1:y2,x1:x2]
        imgCropShape = imgCrop.shape
        aspectRatio = h / w




        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            #prediction, index = classifier.getPrediction(imgWhite)

            #print(prediction, index)


        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

            #prediction, index = classifier.getPrediction(imgWhite) 
        #传递imgWhite，imgCrop出错率太高了，因为收集数据集也是收集的imgWhite
        img1 = data_transform(Image.fromarray(imgWhite))

        img2 = torch.unsqueeze(img1, dim=0)







        with torch.no_grad():

            output = torch.squeeze(model(img2.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        index=class_indict[str(predict_cla)]
        prediction=predict[predict_cla].numpy()

        cv2.putText(imgOutput, index, (x, y-30 ), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset),
                    (x + w+offset, y + h+offset), (255, 0, 255), 4)
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

        print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())

        # for i in range(len(predict)):
        #     print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
        #                                           predict[i].numpy()))
    cv2.imshow("Image", imgOutput)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()



