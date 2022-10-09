import matplotlib.pyplot as plt
import numpy as np
import math
def example():
    plt.figure()
    theta = np.linspace(0,2*math.pi,100)
    r = 3*np.ones(100)
    plt.plot(theta,r,color='b')
    plt.show()
def subplot():
    #plot 1:
    xpoints = np.array([0, 6])
    ypoints = np.array([0, 100])

    plt.subplot(1, 2, 1)#一行两列第一图，初始化一个Axes，并默认一个figure？大概
    plt.plot(xpoints,ypoints)#在上Axes绘图
    plt.title("plot 1")

    #plot 2:
    x = np.array([1, 2, 3, 4])
    y = np.array([1, 4, 9, 16])

    plt.subplot(1, 2, 2)#一行两列第二图，初始化一个Axes？大概
    plt.plot(x,y)#在上Axes绘图
    plt.title("plot 2")

    plt.suptitle("RUNOOB subplot Test")#figure标题，大概
    plt.show()

def subplots():
    # 创建一些测试数据
    x = np.linspace(0, 2*np.pi, 400)
    y = np.sin(x**2)

    # 创建一个画像和子图 -- 图1
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title('Simple plot')

    # 创建两个子图 -- 图2
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot(x, y)
    ax1.set_title('Sharing Y axis')
    ax2.scatter(x, y)

    # # 创建四个子图 -- 图4
    # fig, axs = plt.subplots(2, 2, subplot_kw=dict(projection="polar"))
    # axs[0, 0].plot(x, y)
    # axs[1, 1].scatter(x, y)
    #
    # # 共享 x 轴
    # plt.subplots(2, 2, sharex='col')
    #
    # # 共享 y 轴
    # plt.subplots(2, 2, sharey='row')
    #
    # # 共享 x 轴和 y 轴
    # plt.subplots(2, 2, sharex='all', sharey='all')
    #
    # # 这个也是共享 x 轴和 y 轴
    # plt.subplots(2, 2, sharex=True, sharey=True)
    #
    # # 创建标识为 10 的图，已经存在的则删除
    # fig, ax = plt.subplots(num=10, clear=True)

    plt.show()

def example_坐标系对象数组():
    theta = np.linspace(0,2*math.pi,100)
    r = np.ones(100)

    plt.figure()#创建画布
    axs= {}#创建坐标系对象的数组，使用[]报错

    for i in range(9):#以下行为皆在上行打开的figure中展开，大概

        axs[i]=plt.subplot(111,projection='polar')#将创造的极坐标系存为对象

        axs[i].scatter(1*i,1*i,color='k')#使用对象名与方法操作对象
        axs[i].scatter(5*i,5*i,color='k')
        axs[i].scatter(10*i,10*i,color='k')
        axs[i].plot(theta,2*r,color='b')

    plt.show()#展示figure

def hh():
    theta = np.linspace(0, 2 * math.pi, 100)
    r = np.ones(100)
    plt.ion()#打开交互模式
    plt.figure()  # 创建画布
    axs = {}  # 创建坐标系对象的数组，使用[]报错

    for i in range(9):  # 以下行为皆在上行打开的figure中展开，大概
        #plt.clf()#清理figure
        plt.cla()#清理Axes
        #由于每次都是在新的Axes上绘制，显示，故cla做一个清一个，clf做一个清全部小题大做
        axs[i] = plt.subplot(111, projection='polar')  # 将创造的极坐标系存为对象


        axs[i].scatter(1 * i, 1 * i, color='k')  # 使用对象名与方法操作对象
        axs[i].scatter(5 * i, 5 * i, color='k')
        axs[i].scatter(10 * i, 10 * i, color='k')
        axs[i].plot(theta, 2 * r, color='b')

        plt.pause(1)#暂停一秒
    plt.ioff()#关闭交互模式

hh()
