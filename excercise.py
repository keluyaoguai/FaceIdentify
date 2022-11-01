import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import imageio
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

def example_donghua():
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
def ex_animation():
    fig, ax = plt.subplots()
    y1 = []
    tmp = []
    for i in range(10):
        y1.append(i)  # 每迭代一次，将i放入y1中画出来
        temp = ax.bar(y1, height=y1, width=0.3)
        tmp.append(temp)
        temp = ax.plot(y1,y1)
        tmp.append(temp)

    ani = animation.ArtistAnimation(fig, tmp, interval=1, repeat_delay=1000)#将在figure中的，每次绘制的坐标系图像数组逐个播放
    ani.save("柱状图.gif", writer='pillow')

def ex_gif2():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    fig = plt.figure()
    # 设置画布宽高（坐标值）
    plt.xlim(0, 11)
    plt.ylim(0, 20)

    artists = []#动画帧数组
    # 总共10帧，每帧10个点
    for i in range(10):
        frame = []#帧元素数组
        for j in range(10):
            frame += plt.plot(x[j], y[j] + i, "o")  # 注意这里要+=，对列表操作而不是appand
            artists.append(frame)
    #保存为GIF
    ani = animation.ArtistAnimation(fig=fig, artists=artists, repeat=False, interval=10)
    ani.save('2.gif', fps=30)

def example_imageio():

    fig = plt.figure(figsize=(5,5))
    for i in range(0,40):
        x = np.linspace(0,10,100)
        y = 10*np.sin(i*np.pi/10)*np.sin(x)
        plt.clf()
        plt.xlim((0,2*np.pi))
        plt.ylim((-12,12))
        plt.plot(x,y,color='r')
        plt.savefig(r"D:/PyCharm Community Edition 2022.2.2/Z-code/FaceIdentify/image/"+"figure"+str(i)+".jpg")#路径尾/image/要加一个/不然最后的一段会被识别为文件名
    gif_images = []
    for i in range(0,40):
        gif_images.append(imageio.imread(r"D:/PyCharm Community Edition 2022.2.2/Z-code/FaceIdentify/image/"+"figure"+str(i)+".jpg"))
        imageio.mimsave('GIF.gif',gif_images,fps =20)

def example_():
    gif_frame = []
    for i in range(0, 200):
        gif_frame.append(imageio.imread(r"D:/PyCharm Community Edition 2022.2.2/Z-code/FaceIdentify/image_2/" + "figure" + str(i) + ".jpg"))
        imageio.mimsave("solarSystem_2.gif", gif_frame, fps=15)

def str查询():
    str = '九天雷霆双脚蹬'
    str1='电光毒龙钻'
    str2='霹雳布袋雷霆戏'
    if '雷霆' in str:
        print(1)
    if '雷霆' in str1:
        print(2)
    if '雷霆' in str2:
        print(3)
def 爬虫():
    import time
    import requests
    import re
    from lxml import etree
    def severtan(strs):
        api = "https://sc.ftqq.com/SCT179731TCl32tfNhYrAROdNoUIJvboGd.send"
        title = u"乌克兰相关"
        data = {"text": title, "desp": strs}
        req = requests.post(api, data=data)

    # 爬虫流程
    # 第一步-确定要爬的网址url
    # 第二步-确定要爬的数据再网页内的地址path
    # 第三步-通过requests.get(url)下载url的内容到变量lmth
    # 第四步-通过etree.HTML(lmth)将lmth内容转码（？）至变量pages
    # 第五步-通过 变量.xpath(path)将变量内符合path的数据存入列表
    ur2 = 'https://xnews.jin10.com/page/2'  # 网址的地址字符串
    # requests.get()生成的是一个response对象有如下属性
    # 1.r.status_code： HTTP请求的返回状态，200表示连接成功，404表示失败
    # 2. r.text： HTTP响应内容的字符串形式，即，ur对应的页面内容
    # 3. r.encoding：从HTTP header中猜测的响应内容编码方式
    # 4. r.apparent_encoding：从内容中分析出的响应内容编码方式（备选编码方式）
    # 5. r.content： HTTP响应内容的二进制形式
    html = requests.get(ur2)  # 该对象储存着ur2网址的网页的所有内容
    titleList = []
    while 1:
        for i in range(10):  # range内的数字确定查询页数，与ur2无关
            new_ur2 = re.sub('page/\d+', 'page/%d' % (i + 1), ur2)
            html = requests.get(new_ur2)
            # html.encoding = "utf-8"

            aimPath = '//*[@class="jin10-news-list-item-info"]/a/p/text()'  # 一个网页的某类内容的地址字符串
            pages = etree.HTML(html.text)  # 将html内的文本转码存至pages？大概
            # 当内填html.content时，内容为乱码
            aimList = pages.xpath(aimPath)  # 从pages提取所有xpath路径出的数据并存入title列表？

            for each in aimList:
                if '乌克兰' in each:
                    if each not in titleList:
                        print(each)  # 这里会先输出这一批次最新的新闻
                        # sever酱推送
                        # severtan(each)
                        titleList.append(each)  # 防止重复推送

        # 每十分钟查询最新的十页一次
        time.sleep(600)

def zip网址下载():
    import requests
    import zipfile
    import tempfile

    url = "https://mapopen-website-wiki.bj.bcebos.com/static_zip/BaiduMap_cityCenter.zip"  # 资源的网址
    def get_data(url):
        response = requests.get(url)#下载资源？
        return response.content#返回网址以及资源？

    #将用requests.get下载的资源存到data变量
    data = get_data(url)  # data为byte字节
    print(type(data))
    # _tmp_file = tempfile.TemporaryFile()  # 创建临时文件
    # print(_tmp_file)
    # #zip格式的数据写入临时文件
    # _tmp_file.write(data)  # byte字节数据写入临时文件
    #
    # zf = zipfile.ZipFile(_tmp_file, mode='r')
    # for names in zf.namelist():
    #     f = zf.extract(names, './zip')  # 解压到zip目录文件下
    #     print(f)
    # zf.close()
def 文件夹创建():
    import os
    key_word = '斗罗'
    savePath = 'D:/Z -codefile/spider/novel/' + key_word + '/'
    def folder_creat(path):
        folder = os.path.exists(path)

        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
            print
            "---  new folder...  ---"
            print
            "---  OK  ---"

        else:
            print
            "---  There is this folder!  ---"
    folder_creat(savePath)
def hh():
    import urllib.parse
    s = "斗罗"
    u= urllib.parse.quote(s)
    print(u)
hh()
