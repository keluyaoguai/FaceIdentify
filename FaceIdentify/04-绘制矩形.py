#导入cv模块
import cv2 as cv
#读取图片，记得加文件后缀
img = cv.imread('ph1.jpg')
#坐标
x,y,w,h=100,100,100,100
#绘制矩形
cv.rectangle(img,(x,y,x+w,y+h),color=(0,225,0),thickness=1)
#绘制圆形
cv.circle(img,center=(x+w,y+h),radius=100,color=(0,0,225),thickness=10)
#显示
cv.imshow("this",img)
#等待
cv.waitKey(10000)
#释放内存
cv.destroyAllWindows()