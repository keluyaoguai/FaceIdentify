import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import imageio
import random


boundary = [500,500]#地图大小
infectDistance = 3#感染距离
numPerson = 100#人数
themap = np.zeros((boundary[0],boundary[1]))
class human:
    '''身体状态：0/1速度'''
    def __int__(self,themap):
        self.state = 0#状态
        self.speed = 2#速度，矢量或标量
        self.seat=list[0,0] # 位置
        self.themap = themap#地图数组
    def initXY(self,numx,numy):
        self.seat[0] = numx
        self.seat[1] = numy
    def infect(self):#感染判定
        '''感染'''
        if self.state == 0:
            for i in range(-infectDistance,infectDistance):
                for j in range(-infectDistance,infectDistance):
                    if themap[self.seat[0]-i,self.seat[0]-j] == 1:
                        self.state =1
                        break
                break
    def themapUpdate(self,themap):#更新themap
        if self.state == 1:
            themap[self.seat[0],self.seat[1]] = 1
    def updadethemap(self,themap):#内置themap更新
        self.themap = themap
    def move(self):
        self.seat[0] += self.speed*random.randint(-1,1)
        self.seat[1] += self.speed*random.randint(-1, 1)

persons = [human() for _ in range(numPerson)]#创建对象数组
for j in range(0, numPerson):  # 初始化坐标
    a = random.randint(0, boundary[0])
    b = random.randint(0, boundary[0])
    persons[j].initXY(a,b)
persons[random.randint(0,numPerson)].state = 1#随机一个person成为丧尸
plt.ion()  # 打开交互模式
plt.figure()  # 创建画布
def draw():
    px = list[-1]
    py = list[-1]
    zx = list[-1]
    zy = list[-1]
    for j in range(0,numPerson):#创建点坐标集
        if persons[j].state == 0:
            px.append(persons[j].seat[0])
            py.append(persons[j].seat[1])
        elif persons[j].state == 1:
            zx.append(persons[j].seat[0])
            zy.append(persons[j].seat[1])
    plt.cla()  # 清理Axes
    plt.scatter(px,py,color = 'g')
    plt.scatter(zx,zy,color = 'r')
    plt.pause(1)  # 暂停一秒

for i in range(1000):#循环一千次判定
    for j in range(0,numPerson):#将丧尸坐标更新至themap
        persons[j].themapUpdate(themap)
    for j in range(0,numPerson):#将内置themap同步
        persons[j].updadethemap(themap)
    for j in range(0,numPerson):#判定此回合persons是否感染
        persons[j].infect()
    draw()
    for j in range(0,numPerson):#此回合人物移动
        persons[j].move()
    themap = np.zeros((boundary[0], boundary[1]))
plt.ioff()#关闭交互模式
