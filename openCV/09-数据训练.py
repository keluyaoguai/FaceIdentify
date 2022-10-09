import os
import cv2 as cv
from PIL import Image
import numpy as np

def getImageAndLabels(path):
    #储存人脸数据
    facesSamples = []
    #储存姓名数据
    ids = []
    #储存图片信息
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    #加载分类器
    face_detector = cv.CascadeClassifier()
