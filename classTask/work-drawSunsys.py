#python实现线性表抽象数据类型
#if __name__ == "__main__":#当此语句下的代码是主文件身份时运行，导入其他文件后不运行
def practice_class():
    class mysqlist():
        def __init__(self,size):
            self.size=size
            self.sqlist=[]
        def listinsert(self,i,x):#在i处插入x
            if i<1 or i>self.size:#判断i是否合法
                print("Insert Location Error！")
                return False
            else:
                self.sqlist.insert(i,x)#插入
                return True
        def listdelete(self,i):#删除i处
            if i<1 or i>self.size:#判断i是否合法
                print("Delete Location Error！")
                return False
            else:
                self.sqlist.pop(i)#删除i处
                return False
        def findelem(self,i):#查询i处数值
            if i<1 or i>self.size:#判断i是否合法
                print("search Location Error！")
                return False
            else:
                return self.sqlist[i]
        def showlist(self):#查询整体
            return self.sqlist
def practice_matplotlib_1():
    import matplotlib.pyplot as plt
    import numpy as np
    x = np.linspace(-2,2,50)
    y1 = 2*x*x
    y2 = 2*x
    plt.figure(num=1)#以下直至另一个figure的plt操作属于这个figure？大概
    plt.plot(x,y1)#根据坐标绘图，与上一行figure绑定

    plt.figure(num=2)#(num,figsize,dpi, facecolor,edgecolor,frameon,FigureClass,clear）
    plt.plot(x,y2,color='red',linewidth=1,linestyle=':')#()
    plt.plot(x,y1)

    plt.xlim((-3,3))#限制的是初始视野，不是绘制区间
    plt.ylim((-3,3))#limit限制

    plt.xlabel('x')#给xy轴重命名
    plt.ylabel('y')#label标签

    new_ticks = np.linspace(0,2,5)
    plt.xticks(new_ticks)#将xy轴的坐标重命名
    plt.yticks([1,3,4],[r'$bad$',r'$good\ \alpha$','perfact'])#$转换字体  \显示空格与希腊字母

    plt.show()#show一次展示所有figure


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import imageio

class heavenlyBodies():
    '''天体半径，轨迹半径，初始角度，公转周期，颜色'''
    def __init__(self,size,r,o,time,color):
        self.size = size*20
        self.xy = [int(r*5),o]
        self.speed = ((2*math.pi)/time)
        self.color = color

    def move(self):
        self.xy[1]+= self.speed
        # while self.xy[1]>=2*math.pi:#极坐标可以适应大于2pi的数，无需处理
        #     self.xy[1]-=2*math.pi

    def draw(self):
        action = [1,2,3]
        #action[0] = solar_system.scatter(self.xy[1],self.xy[0],self.size,self.color)#绘制天体
        solar_system.scatter(self.xy[1], self.xy[0], self.size, self.color)  # 绘制天体
        r = self.xy[0] * np.ones(50)  # 轨道半径数据
        therta = np.linspace(0, 2 * math.pi,50)  # 轨道角度数据
        #action[1] = solar_system.plot(therta, r,linewidth=1,color=body_color[i])#绘制轨道
        solar_system.plot(therta, r, linewidth=1, color=self.color)  # 绘制轨道
        #action[2] = solar_system.scatter(star_x, (star_x + 200), s=star_r, c='white', alpha=star_l)  # 显示星星
        solar_system.scatter(star_x, (star_x + 200), s=star_r, c='white', alpha=star_l)  # 显示星星
        #return action
def body_data():
    global body_radius
    global orbit_radius
    global orbit_time
    global body_color
    global orbit_theta
    global star_x
    global star_r
    global star_l
    #太阳系天体半径适当化：太阳20水星1金星3地球3火星2木星7土星6天王星5海王星5
    body_radius = [50,1,2.4,2.6,1.4,23,19,7,7]
    #太阳系天体轨道半径适当化：太阳0水星4金星8地球10火星15木星20土星30天王星40海王星50
    orbit_radius = [0,20,26,32,44,64,94,132,174]
    #太阳系天体公转周期适当化：太阳0水星80金星200地球400火星800木星4000土星10000天王星30000海王星60000
    orbit_time = [1,80,200,400,800,4000,10000,30000,60000]
    #太阳系天体初始角度适当化
    orbit_theta = 2*math.pi*np.random.rand(9)
    #太阳系天体颜色适当化：太阳 深橙 水星 金黄 金星 金黄 地球 深蓝 火星 黄土赭 木星 秋麒麟 土星 耐火砖 天王星 浅蓝 海王星 蓝
    body_color = ['#FF8C00','#FFD700','#FFD700','#0000FF','#A0522D','#DAA520','#B22222','#00FFFF','#4682B4',]
    #星空背景数据
    star_x = 2000*np.random.rand(1000)
    star_r = np.random.rand(1000)
    star_l = np.random.rand(1000)

#导入数据
body_data()
#创建天体对象
sun_body = []
for i in range(9):
    sun_body.append(heavenlyBodies(body_radius[i],orbit_radius[i],orbit_theta[i],orbit_time[i],body_color[i]))
#动画帧list
gif_frame = []
#开启交互模式
#plt.ion()
#一帧图的绘制

figure_num = 200
for i in range(figure_num):
    frame = []  # 帧元素list
    fig = plt.figure(1, (10, 10), facecolor='k')  # 打开画布 #方法的前的等号不会使方法失效
    plt.clf()
    #plt.cla()
    solar_system = plt.subplot(projection='polar', facecolor='k')  # 切换坐标系为极坐标
    plt.xlim(0, 2 * math.pi)  # x在极坐标下控制角度
    plt.ylim(0, 900)  # y在极坐标系下控制半径
    plt.yticks([])#不显示半径数值
    solar_system.grid(False)  # 不显示格子
    for j in range(9):
        #frame += sun_body[j].draw()#绘制天体
        sun_body[j].draw()  # 绘制天体
        sun_body[j].move()#计算天体下一位置
    #gif_frame.append(frame)
    #plt.pause(0.001)
    plt.savefig(r"D:/PyCharm Community Edition 2022.2.2/Z-code/FaceIdentify/image_2/" + "figure" + str(i) + ".jpg")
#关闭交互模式
#plt.ioff()
#imageio制作gif
for i in range(0,figure_num):
    gif_frame.append(imageio.imread(r"D:/PyCharm Community Edition 2022.2.2/Z-code/FaceIdentify/image_2/"+"figure"+str(i)+".jpg"))
    imageio.mimsave('solarSystem.gif',gif_frame,fps=15)
#animation制作gif
#ani = animation.ArtistAnimation(fig=fig, artists=gif_frame, repeat=True, interval=10)
#ani.save('3.gif')
