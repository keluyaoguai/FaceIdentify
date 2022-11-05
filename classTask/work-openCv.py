import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
#math = '../FaceIdentify/ph1.jpg'
math = '../FaceIdentify/ph10.jfif'
gaosimohu = 5#奇数，控制模糊度

def fushi(img,i,j):
    '''图像，次数，内核大小'''
 # 2. cv2.MORPH_OPEN 先进行腐蚀操作，再进行膨胀操作
    for i in range(i):
        kernel = np.ones((j, j), np.uint8)
        opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    return opening
#返回
def mask(math):
    img = cv.imread(math)
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    #rect = (1, int(0.19 * img.shape[0]), int(0.90 * img.shape[1]), img.shape[0])
    rect = (1, 1, img.shape[1], img.shape[0])# 川普的矩形
    mask,bgdModel,fgdModel = cv.grabCut(img, mask, rect, bgdModel, fgdModel, 20, cv.GC_INIT_WITH_RECT)
    #首先让我们看看矩形模式。我们加载图像，创建一个类似的蒙版图像。
    # 然后运行抓取。它修改了蒙版图像。
    # 在新的掩模图像中，像素将被标记为表示背景/前景的如上所述四个标记。
    # 因此，我们修改掩模，使得所有0像素和2像素都被置为0（即背景），并且所有1像素和3像素被置为255（即前景像素）。
    mask = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
    return mask
#img = img * mask2[:, :, np.newaxis]
img = cv.imread(math)
print(0)
mask = mask(math)
print(-1)
mask = fushi(mask,5,6)
print(1)
mask_inv = cv.bitwise_not(mask)
print(2)
#mask_inv = fushi(mask_inv,5,6)
res = cv.bitwise_and(img, img, mask=mask)
print(3)
background = cv.bitwise_and(img, img, mask = mask_inv)
blur_background = cv.GaussianBlur(background, (gaosimohu, gaosimohu), 0)  # (5, 5)表示高斯矩阵的长与宽都是5，标准差取0
added_img = cv.add(res, blur_background)


cv.imshow("mask2_inv", mask_inv)
cv.imshow("mask", mask)
cv.imshow("res",res)
cv.imshow("background",background)
cv.imshow("added",added_img)
cv.waitKey(0)
cv.destroyAllWindows()
def the():
    img = cv.imread(math)
    #img = cv.cvtColor(img,cv.COLOR_BGR2HSV)
    #img = cv.resize(img,(400,400))
    mask = np.zeros(img.shape[:2], np.uint8)

    # zeros(shape, dtype=float, order='C')，参数shape代表形状，(1,65)代表1行65列的数组，dtype:数据类型，可选参数，默认numpy.float64
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (1,int(0.19*img.shape[0]), int(0.90*img.shape[1]), img.shape[0])#川普的矩形
    #cv.rectangle(img,(rect[0],rect[1],rect[2],rect[3]),color=(0,225,0),thickness=1)
    # 函数原型：grabCut(img, mask, rect, bgdModel, fgdModel, iterCount, mode=None)
    # img - 输入图像
    # mask-掩模图像，用来确定那些区域是背景，前景，可能是前景/背景等。可以设置为：cv.GC_BGD,cv.GC_FGD,cv.GC_PR_BGD,cv.GC_PR_FGD，或者直接输入 0,1,2,3 也行。
    # rect - 包含前景的矩形，格式为 (x,y,w,h)
    # bdgModel, fgdModel - 算法内部使用的数组. 你只需要创建两个大小为 (1,65)，数据类型为 np.float64 的数组。
    # iterCount - 算法的迭代次数
    # mode cv.GC_INIT_WITH_RECT 或 cv.GC_INIT_WITH_MASK，使用矩阵模式还是蒙板模式。
    cv.grabCut(img, mask, rect, bgdModel, fgdModel, 20, cv.GC_INIT_WITH_RECT)

    # np.where 函数是三元表达式 x if condition else y的矢量化版本
    # result = np.where(cond,xarr,yarr)
    # 当符合条件时是x，不符合是y，常用于根据一个数组产生另一个新的数组。
    # | 是逻辑运算符or的另一种表现形式
    #如果mask==2或==0，mask2=0，否则mask2=1
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    # mask2[:, :, np.newaxis] 增加维度
    #数组相乘是每个位置的数字相乘
    #mask增加唯维度后，每个像素相乘是(R,G,B)*(D)=(R*D,G*D,B*D)
    img = img * mask2[:, :, np.newaxis]

    cv.imshow('grabcut',img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    img = cv.imread(math)
    # img = cv.resize(img,(400,400))
    mask = np.zeros(img.shape[:2], np.uint8)

    # zeros(shape, dtype=float, order='C')，参数shape代表形状，(1,65)代表1行65列的数组，dtype:数据类型，可选参数，默认numpy.float64
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (1, int(0.19 * img.shape[0]), int(0.90 * img.shape[1]), img.shape[0])  # 川普的矩形
    # cv.rectangle(img,(rect[0],rect[1],rect[2],rect[3]),color=(0,225,0),thickness=1)
    # 函数原型：grabCut(img, mask, rect, bgdModel, fgdModel, iterCount, mode=None)
    # img - 输入图像
    # mask-掩模图像，用来确定那些区域是背景，前景，可能是前景/背景等。可以设置为：cv.GC_BGD,cv.GC_FGD,cv.GC_PR_BGD,cv.GC_PR_FGD，或者直接输入 0,1,2,3 也行。
    # rect - 包含前景的矩形，格式为 (x,y,w,h)
    # bdgModel, fgdModel - 算法内部使用的数组. 你只需要创建两个大小为 (1,65)，数据类型为 np.float64 的数组。
    # iterCount - 算法的迭代次数
    # mode cv.GC_INIT_WITH_RECT 或 cv.GC_INIT_WITH_MASK，使用矩阵模式还是蒙板模式。
    cv.grabCut(img, mask, rect, bgdModel, fgdModel, 20, cv.GC_INIT_WITH_RECT)

    # np.where 函数是三元表达式 x if condition else y的矢量化版本
    # result = np.where(cond,xarr,yarr)
    # 当符合条件时是x，不符合是y，常用于根据一个数组产生另一个新的数组。
    # | 是逻辑运算符or的另一种表现形式
    # 如果mask==2或==0，mask2=0，否则mask2=1
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    # mask2[:, :, np.newaxis] 增加维度
    # 数组相乘是每个位置的数字相乘
    # mask增加唯维度后，每个像素相乘是(R,G,B)*(D)=(R*D,G*D,B*D)
    img1 = img * mask2[:, :, np.newaxis]
    img2 = img * mask[:, :, np.newaxis]

    cv.imshow('mask2', img1)
    cv.imshow('mask', img2)
    cv.waitKey(0)
    cv.destroyAllWindows()
