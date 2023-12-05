import cv2
import tensorflow
import numpy as np
import math
import pyttsx3
import os

# import matplotlib.pyplot as plt

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

# Loading the Model
model = cv2.dnn_DetectionModel(frozen_model, config_file)
# Loading the labels
classlabels = []
file_name = 'coco.txt'
with open(file_name, 'rt') as fpt:
    classlabels = fpt.read().rstrip('\n').split('\n')

engine = pyttsx3.init()

# Configuring my model according to the given req in the config file
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean([127.5, 127.5, 127.5])
model.setInputSwapRB(True)

font_scale = 1
font = cv2.FONT_HERSHEY_PLAIN

threshold = 310720

# img = cv2.imread('C:/My Files/Spyder/ODB/1.jpg')
def getObjects(img, objects=[]):
    classIndex, confidence, bbox = model.detect(img, confThreshold=0.5, nmsThreshold=0.2)
    if len(objects) == 0: objects = classlabels
    objectInfo = []
    if len(classIndex) != 0:
        for ClassID, conf, boxes in zip(classIndex.flatten(), confidence.flatten(), bbox):
            className = classlabels[ClassID - 1]
            if className in objects:
                objectInfo.append([boxes, className])
                print(className)
                print(boxes)
                cv2.rectangle(img, boxes, (255, 0, 0), 2)
                cv2.putText(img, classlabels[ClassID - 1], (boxes[0] + 10, boxes[1] + 40), font, fontScale=font_scale,
                            color=(0, 255, 0), thickness=1)
                centx = int(boxes[0]*img.shape[1])
                centy = int(boxes[1]*img.shape[0])
                dist = math.sqrt((centx - img.shape[1]/2)**2 + (centy - img.shape[0]/2)**2)
                print(dist)
                if dist < threshold:
                    print("voice")
                    engine.say(f"{classlabels[ClassID - 1]} at {int(dist*2)/1000} centimeters.")
                    engine.runAndWait()
    return img, objectInfo


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    while True:
        success, img = cap.read()
        result = getObjects(img, objects=[])
        cv2.imshow('img', img)
        cv2.waitKey(1)
