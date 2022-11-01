#导入opencv模块
import cv2 as cv
#捕获摄像头
cap=cv.VideoCapture(0)

flag=1
num=1

while(cap.isOpened()):#检测是否在开启状态
    ret_flag,Vshow = cap.read()#得到每帧图像
    cv.imshow("capture_test",Vshow)#显示图像
    #k等于循环时间段中键盘的输入
    k = cv.waitKey(1)
    #按s捕获图像
    if k == ord('s'):
        cv.imwrite("D:/Z -codefile/face-identify/imageData/hsl/"+str(num)+".hsl"+".jpg",Vshow)#储存图像,路径不能含中文
        #cv.imwrite("D:/Z -codefile/face-identify/imageData/hsl/" + str(num) + ".hsl" + ".jpg", Vshow)
        #cv.imwrite("D:/Z -codefile/face-identify/imageData/hmy/" + str(num) + ".hmy" + ".jpg", Vshow)
        #cv.imwrite("D:/Z -codefile/face-identify/imageData/hmr/" + str(num) + ".hmr" + ".jpg", Vshow)

        print("success to save"+str(num)+'.name'+".jpg")
        print("-------------------------")
        num += 1#作为文件后缀
    #按空格推出
    elif k == ord(' '):
        break
#释放摄像头
cap.release()
#释放内存
cv.destroyAllWindows()
