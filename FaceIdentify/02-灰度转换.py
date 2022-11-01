#导入cv模块
import cv2 as cv
#读取图片，记得加文件后缀
img = cv.imread('ph1.jpg')
#灰化图片
gary_img = cv.cvtColor(img,cv.COLOR_YCrCb2RGB)
#显示图片
cv.imshow('read_img',gary_img)
#等待
cv.waitKey(0)
#释放内存
cv.destroyAllWindows()