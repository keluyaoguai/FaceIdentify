# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
#
# img = cv2.imread('../FaceIdentify/ph5.jpg')
# img = cv2.resize(img,(400,400))
# mask = np.zeros(img.shape[:2], np.uint8)
#
# # zeros(shape, dtype=float, order='C')，参数shape代表形状，(1,65)代表1行65列的数组，dtype:数据类型，可选参数，默认numpy.float64
# bgdModel = np.zeros((1, 65), np.float64)
# fgdModel = np.zeros((1, 65), np.float64)
# rect = (1, 1, img.shape[1], img.shape[0])
# # 函数原型：grabCut(img, mask, rect, bgdModel, fgdModel, iterCount, mode=None)
# # img - 输入图像
# # mask-掩模图像，用来确定那些区域是背景，前景，可能是前景/背景等。可以设置为：cv2.GC_BGD,cv2.GC_FGD,cv2.GC_PR_BGD,cv2.GC_PR_FGD，或者直接输入 0,1,2,3 也行。
# # rect - 包含前景的矩形，格式为 (x,y,w,h)
# # bdgModel, fgdModel - 算法内部使用的数组. 你只需要创建两个大小为 (1,65)，数据类型为 np.float64 的数组。
# # iterCount - 算法的迭代次数
# # mode cv2.GC_INIT_WITH_RECT 或 cv2.GC_INIT_WITH_MASK，使用矩阵模式还是蒙板模式。
# cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 20, cv2.GC_INIT_WITH_RECT)
# print(2)
# # np.where 函数是三元表达式 x if condition else y的矢量化版本
# # result = np.where(cond,xarr,yarr)
# # 当符合条件时是x，不符合是y，常用于根据一个数组产生另一个新的数组。
# # | 是逻辑运算符or的另一种表现形式
# mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
# # mask2[:, :, np.newaxis] 增加维度
# img = img * mask2[:, :, np.newaxis]
#
# cv2.imshow('grabcut',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import numpy as np
import math
def panelAbstract(srcImage):
    #   read pic shape
    imgHeight,imgWidth = srcImage.shape[:2]
    imgHeight = int(imgHeight);imgWidth = int(imgWidth)
    # 均值聚类提取前景:二维转一维
    imgVec = np.float32(srcImage.reshape((-1,3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    ret,label,clusCenter = cv2.kmeans(imgVec,2,None,criteria,10,flags)
    clusCenter = np.uint8(clusCenter)
    clusResult = clusCenter[label.flatten()]
    imgres = clusResult.reshape((srcImage.shape))
    imgres = cv2.cvtColor(imgres,cv2.COLOR_BGR2GRAY)
    bwThresh = int((np.max(imgres)+np.min(imgres))/2)
    _,thresh = cv2.threshold(imgres,bwThresh,255,cv2.THRESH_BINARY_INV)
    threshRotate = cv2.merge([thresh,thresh,thresh])
    # 确定前景外接矩形
    #find contours
    #imgCnt,contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    minvalx = np.max([imgHeight,imgWidth]);maxvalx = 0
    minvaly = np.max([imgHeight,imgWidth]);maxvaly = 0
    maxconArea = 0;maxAreaPos = -1
    for i in range(len(contours)):
        if maxconArea < cv2.contourArea(contours[i]):
            maxconArea = cv2.contourArea(contours[i])
            maxAreaPos = i
    objCont = contours[maxAreaPos]
    # 旋转校正前景
    rect = cv2.minAreaRect(objCont)
    for j in range(len(objCont)):
        minvaly = np.min([minvaly,objCont[j][0][0]])
        maxvaly = np.max([maxvaly,objCont[j][0][0]])
        minvalx = np.min([minvalx,objCont[j][0][1]])
        maxvalx = np.max([maxvalx,objCont[j][0][1]])
    if rect[2] <=-45:
        rotAgl = 90 +rect[2]
    else:
        rotAgl = rect[2]
    if rotAgl == 0:
        panelImg = srcImage[minvalx:maxvalx,minvaly:maxvaly,:]
    else:
        rotCtr = rect[0]
        rotCtr = (int(rotCtr[0]),int(rotCtr[1]))
        rotMdl = cv2.getRotationMatrix2D(rotCtr,rotAgl,1)
        imgHeight,imgWidth = srcImage.shape[:2]
        #图像的旋转
        dstHeight = math.sqrt(imgWidth *imgWidth + imgHeight*imgHeight)
        dstRotimg = cv2.warpAffine(threshRotate,rotMdl,(int(dstHeight),int(dstHeight)))
        dstImage = cv2.warpAffine(srcImage,rotMdl,(int(dstHeight),int(dstHeight)))
        dstRotimg = cv2.cvtColor(dstRotimg,cv2.COLOR_BGR2GRAY)
        _,dstRotBW = cv2.threshold(dstRotimg,127,255,0)
        imgCnt,contours, hierarchy = cv2.findContours(dstRotBW,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        maxcntArea = 0;maxAreaPos = -1
        for i in range(len(contours)):
            if maxcntArea < cv2.contourArea(contours[i]):
                maxcntArea = cv2.contourArea(contours[i])
                maxAreaPos = i
        x,y,w,h = cv2.boundingRect(contours[maxAreaPos])
        #提取前景：panel
        panelImg = dstImage[int(y):int(y+h),int(x):int(x+w),:]

    return panelImg

if __name__=="__main__":
   srcImage = cv2.imread('../FaceIdentify/ph5.jpg')
   a=panelAbstract(srcImage)
   cv2.imshow('figa',a)
   cv2.waitKey(0)
   cv2.destroyAllWindows()


