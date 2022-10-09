#导入cv模块
import cv2 as cv
#定义识别函数
def face_detect_demo(img):
    # 灰化图像
    gary_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #确定识别函数使用的模型(脸、眼、嘴)
    face_detect = cv.CascadeClassifier('D:/Program/ZZZ\opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
    #face_detect = cv.CascadeClassifier('D:/Program/ZZZ\opencv/sources/data/haarcascades/haarcascade_eye.xml')
    #face_detect = cv.CascadeClassifier('D:/Program/ZZZ\opencv/sources/data/haarcascades/haarcascade_smile.xml')
    #识别位置，存入二维数组face，是所有的位置
    face = face_detect.detectMultiScale(gary_img)#(图像，搜索框缩放比例，结果怀疑次数，默认0，（筛掉，比这小的），（筛掉，比大）)
    #在原图中添加矩形
    for x,y,w,h in face:#for结构已经起到了多人标记的功能
        cv.rectangle(img,(x,y),(x+w,y+h),color=(0,255,0),thickness=2)
    #显示结果图像
    cv.imshow("result",img)

#调用摄像头
#cap = cv.VideoCapture('E:/视频/假面骑士/帝骑-decade/15.mkv')#数字调用摄像头‘视频地址’调用视频
#cap = cv.VideoCapture('hmy1.mp4')
cap = cv.VideoCapture(0)

#等待
while True:
    flag,frame = cap.read()
    if not flag:
        break
    face_detect_demo(frame)
    if ord(' ') == cv.waitKey(0):
        break
#释放内存
cv.destroyAllWindows()
#释放摄像头
cap.release()