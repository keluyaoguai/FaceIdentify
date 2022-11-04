import numpy as np
import cv2 as cv

# 加载彩色灰度图像
# img = cv.imread('zzumap.png',1)
# img = cv.resize(img, (1020, 768))
# cv.namedWindow("image",cv.WINDOW_AUTOSIZE)
# cv.imshow('image',img)
# cv.waitKey(0)
# cv.destroyAllWindows()
#第一段程序结束，关于在图片上增加滚动条请参考
# https://blog.csdn.net/qq_34801642/article/details/86595698?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.add_param_isCf&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.add_param_isCf

#第二段程序说明存储格式
# img = cv.imread('test.jpg')
# type(img)
# print(img.shape)
#
# img[0]
# img[1]
#存储格式结束

#继续读取特定像素 适合
# img = cv.imread('messi5.jpg')#img.shape
# px = img[100,100] #第100行第100列像素点
# print(px)
# blue = img[100,100,0]#第100行第100列像素点的blue值
# print( blue )
# #You can modify the pixel values the same way.
# img[100,100] = [255,255,255]
# print( img[100,100] )
# #Better pixel accessing and editing method :
# # accessing RED value
# img.item(10,10,2)
# # modifying RED value
# img.itemset((10,10,2),100)
# img.item(10,10,2)
# #Accessing Image Properties
# print( img.shape )
# print( img.size )
# print(img.dtype)

#测试兴趣区域 Image ROI
# img = cv.imread('messi5.jpg')
# # ball = img[280:340, 330:390]
# # img[273:333, 100:160] = ball
# grass = img[273:333, 40:100]
# img[273:333, 100:160] = grass
# cv.namedWindow("image",cv.WINDOW_AUTOSIZE)
# cv.imshow('image',img)
# cv.waitKey(0)
# cv.destroyAllWindows()
#测试结束，这段代码有错误待查找

#下面是查找过程 220:280, 270:330
# img = cv.imread('messi5.jpg')
# ball = img[60:120, 170:230]
# # print(img.shape)
# img[100:160, 270:330] = ball
# cv.namedWindow("image",cv.WINDOW_AUTOSIZE)
# cv.namedWindow("image1",cv.WINDOW_AUTOSIZE)
# cv.imshow('image1',ball)
# cv.imshow('image',img)
# cv.waitKey(0)
# cv.destroyAllWindows()
#查找结束

#Splitting and Merging Image Channels
# img = cv.imread('messi5.jpg')
# b,g,r = cv.split(img)
# img = cv.merge((b,g,r))
# img[:,:,2] = 0
# cv.namedWindow("image",cv.WINDOW_AUTOSIZE)
# cv.imshow('image',img)
# cv.waitKey(0)
# cv.destroyAllWindows()

#代码结束
image=cv.imread('photo.jpg')
image = cv.resize(image,None,fx=0.5,fy=0.5)
hsv = cv.cvtColor(image,cv.COLOR_BGR2HSV)
lower_red = np.array([80,70,150])
upper_red = np.array([150,255,255])
heibai = cv.inRange(hsv, lower_red, upper_red)
k=np.ones((5,5),np.uint8)
r=cv.morphologyEx(heibai,cv.MORPH_CLOSE,k)
cv.imshow('image',image)
#颜色替换
rows,cols,channels = image.shape
for i in range(rows):
  for j in range(cols):
    if r[i,j]==255: # 像素点为255表示的是白色，我们就是要将白色处的像素点，替换为红色
      image[i,j]=(0,0,255) # 此处替换颜色，为BGR通道，不是RGB通道
# #新图显示
cv.imshow('red',image)
cv.imshow('hsv',hsv)
cv.imshow('heibai',heibai)
cv.imshow('r',r)
cv.waitKey(0)
cv.destroyAllWindows()
#程序演示结束