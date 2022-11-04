#import the libraries
import cv2 as cv
import numpy as np

#read the image
#录入这个图像
img = cv.imread("the1.jpg")
#convert the BGR image to HSV colour space
#将这个BGR图像转化为HSV空间
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
#obtain the grayscale image of the original image
#获得初始图像的灰度图
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#set the bounds for the red hue
#设置红色色调的边界
lower_red = np.array([160,100,50])
upper_red = np.array([180,255,255])

#create a mask using the bounds set
#使用这个边界与HSV图像创造一个蒙版
mask = cv.inRange(hsv, lower_red, upper_red)
#create an inverse of the mask
#创造这个蒙版的反向态
mask_inv = cv.bitwise_not(mask)
#Filter only the red colour from the original image using the mask(foreground)
#用蒙版从初始图像中过滤出红色，作为前景
res = cv.bitwise_and(img, img, mask=mask)
#Filter the regions containing colours other than red from the grayscale image(background)
#用反向蒙版从灰度图中过滤出除了红色以外的含有颜色的区域，作为背景
background = cv.bitwise_and(gray, gray, mask = mask_inv)
#convert the one channelled grayscale background to a three channelled image
#转化单通道灰度图为三通道图
background = np.stack((background,)*3, axis=-1)
#add the foreground and the background
#将前景与背景叠加
added_img = cv.add(res, background)

#create resizable windows for the images
#为图像创造可调整大小的窗口
cv.namedWindow("res", cv.WINDOW_NORMAL)
cv.namedWindow("hsv", cv.WINDOW_NORMAL)
cv.namedWindow("mask", cv.WINDOW_NORMAL)
cv.namedWindow("added", cv.WINDOW_NORMAL)
cv.namedWindow("back", cv.WINDOW_NORMAL)
cv.namedWindow("mask_inv", cv.WINDOW_NORMAL)
cv.namedWindow("gray", cv.WINDOW_NORMAL)

#display the images
#展示这些图像
cv.imshow("back", background)
cv.imshow("mask_inv", mask_inv)
cv.imshow("added",added_img)
cv.imshow("mask", mask)
cv.imshow("gray", gray)
cv.imshow("hsv", hsv)
cv.imshow("res", res)

if cv.waitKey(0):
    cv.destroyAllWindows()