#导入cv模块
import cv2 as cv
#读取图片，记得加文件后缀
img = cv.imread('ph1.jpg')
#修改尺寸
resize_img = cv.resize(img,dsize=(200,200))
#显示原图

#显示修改图
cv.imshow('this',resize_img)
#打印修改后大小
print("打印前",img.shape)
print("打印后",resize_img.shape)
#等待
while True:
    if 112 == cv.waitKey(0):
        break
#释放内存
cv.destroyAllWindows()