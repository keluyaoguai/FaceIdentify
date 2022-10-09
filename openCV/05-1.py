#导入cv模块
import cv2 as cv
#自定义等比例缩放函数
def img_resize(image):
    height, width = image.shape[0], image.shape[1]
    # 设置新的图片分辨率框架
    width_new = 740
    height_new = 360
    # 判断图片的长宽比率
    if width / height >= width_new / height_new:
        img_new = cv.resize(image, (width_new, int(height * width_new / width)))
    else:
        img_new = cv.resize(image, (int(width * height_new / height), height_new))
    return img_new


#定义识别函数
def face_detect_demo():
    # 灰化图像
    gary_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #调用训练集定义函数？
    body_detect = cv.CascadeClassifier('D:/Program/ZZZ\opencv/sources/data/haarcascades/haarcascade_upperbody.xml')
    #识别位置，存入数组face
    upperbody = body_detect.detectMultiScale(gary_img)
    #在原图中添加矩形
    for x,y,w,h in upperbody:
        cv.rectangle(img,(x,y),(x+w,y+h),color=(0,255,0),thickness=2)
    #显示结果图像
    cv.imshow("result",img)


#读取图像
img1= cv.imread("ph2.jpg")
#修改尺寸
img = img_resize(img1)
#自定义识别、画矩形、输出函数
face_detect_demo()
#等待
cv.waitKey(10000)
#释放内存
cv.destroyAllWindows()