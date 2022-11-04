import cv2 as cv
import numpy as np
#调用摄像头
#cap = cv.VideoCapture(0, cv.CAP_DSHOW)
cap = cv.VideoCapture('E:/视频/假面骑士/帝骑-decade/15.mkv')
#打开默认摄像头？也许
#cap.open(0)

while cap.isOpened():
    flag, frame = cap.read()
    if not flag:
        break
    key_pressed = cv.waitKey(60)
    print('键盘上输入的是', key_pressed)
    frame = cv.resize(frame, (600, 600))
    #frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)#帖子说是canny需要灰度图，但不需要也可以，毕竟是九年前的帖子，opencv应该是变化了
    #边缘检测函数
    frame = cv.Canny(frame, 100, 200)
    #从效果上似乎没有作用，也许也是因为九年间的变化？
    #frame2 = np.dstack((frame, frame, frame))
    cv.namedWindow("windowName",0)
    cv.imshow('my computer', frame)
    #cv.imshow('my computer2', frame2)

    if key_pressed == 27:
        break

cap.release()

cv.destroyAllWindows()