import time
import requests
import re
from lxml import etree
def severtan(strs):
    #api = "https://sc.ftqq.com/SCT179731TCl32tfNhYrAROdNoUIJvboGd.send"
    title = u"乌克兰相关"
    data = {"text":title,"desp":strs}
    req = requests.post(api,data = data)
#爬虫流程
#第一步-确定要爬的网址url
#第二步-确定要爬的数据再网页内的地址path
#第三步-通过requests.get(url)下载url的内容到变量lmth
#第四步-通过etree.HTML(lmth)将lmth内容转码（？）至变量pages
#第五步-通过 变量.xpath(path)将变量内符合path的数据存入列表
ur2 = 'https://xnews.jin10.com/page/2'#网址的地址字符串
# requests.get()生成的是一个response对象有如下属性
# 1.r.status_code： HTTP请求的返回状态，200表示连接成功，404表示失败
# 2. r.text： HTTP响应内容的字符串形式，即，ur对应的页面内容
# 3. r.encoding：从HTTP header中猜测的响应内容编码方式
# 4. r.apparent_encoding：从内容中分析出的响应内容编码方式（备选编码方式）
# 5. r.content： HTTP响应内容的二进制形式
html = requests.get(ur2)#该对象储存着ur2网址的网页的所有内容
titleList = []
while 1:
    for i in range(10):#range内的数字确定查询页数，与ur2无关
        new_ur2 = re.sub('page/\d+','page/%d' % (i+1),ur2)
        html = requests.get(new_ur2)
        #html.encoding = "utf-8"

        aimPath = '//*[@class="jin10-news-list-item-info"]/a/p/text()'#一个网页的某类内容的地址字符串
        pages = etree.HTML(html.text)#将html内的文本转码存至pages？大概
        #当内填html.content时，内容为乱码
        aimList = pages.xpath(aimPath)#从pages提取所有xpath路径出的数据并存入title列表？

        for each in aimList:
            if '乌克兰' in each:
                if each not in titleList:
                    print(each)#这里会先输出这一批次最新的新闻
                    #sever酱推送
                    #severtan(each)
                    titleList.append(each)#防止重复推送

    #每十分钟查询最新的十页一次
    time.sleep(600)

