import time
import requests
import re
from lxml import etree
import os
import urllib.parse
#//*选取文档中的所有元素
#//*[@what]选取文档中的所有元素中拥有what属性的部分
#控制搜索内容
key_word = input("请输入小说关键词\n")
#控制下载页数
start_page = int(input("请输入开始页数\n"))
keyWord_url= urllib.parse.quote(key_word)
novel_namePath = '//*[@class="book_textList2 bd"]/li/a/text()'  # 该关键词搜索内容的小说名
pages_numPath = '//*[@class="pagination clearfix"]/span/text()'#该关键词搜索内容的总页数
pages_numPath = '//div[@class="pagination clearfix"]/span/text()'#该关键词搜索内容的总页数
searchResult = 'http://downnovel.com/search.htm?keyword='+keyWord_url+'&pn=1'#初始网页
#searchResult = 'http://downnovel.com/search.htm?keyword=%E6%96%97%E7%BD%97&pn=1'#初始网页
zipPath = '//a[@class="btn_b"]/@href'#zip资源的网址
theUrl = 'http://downnovel.com'
savePath = 'E:/spider_file/novel/'+key_word+'/'


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
def get_data(url):
    response = requests.get(url)  # 下载资源？
    return response.content  # 返回网址以及资源？
# 将用requests.get下载的资源存到data变量
def save_data(url,savePath,name,type):
    '''资源网址，保存的文件夹路径，文件名称，文件类型(前面有点)'''
    data = get_data(url)  # data为byte字节
    f = open(savePath+name+type, mode='wb')
    f.write(data)
    f.close()

#获取总页数
check = requests.get(searchResult)
firstPage = etree.HTML(check.text)
pagesNum = firstPage.xpath(pages_numPath)  # 这里是一页内小说的网址的列表
pagenum = int(pagesNum[0].split(r"/")[1])

howMany =0
for i in range(start_page-1,pagenum):#爬全部页数的
#for i in range(1):#爬指定页数的
    next_searchResult = re.sub('&pn=\d+', '&pn=%d' % (i + 1), searchResult)
    #下载网页数据
    thePage = requests.get(next_searchResult)#该对象储存着该网页的所有内容
    the_page = etree.HTML(thePage.text)  # 将html内的文本转码存至pages？大概
    #提取内容
    novel_urlPath = '//*[@class="book_textList2 bd"]/li/a/@href'  # 该关键词搜索内容的novel_url
    novel_nameList = the_page.xpath(novel_namePath)#这里是一页内小说的名字的列表
    novel_urlList = the_page.xpath(novel_urlPath)#这里是一页内小说的网址的列表


    for name, url in zip(novel_nameList, novel_urlList):
        thisNovel = theUrl + url#小说网址
        #print(thisNovel)
        # 下载网页数据
        thisPage = requests.get(thisNovel)  # 该对象储存着该网页的所有内容
        this_page = etree.HTML(thisPage.text)  # 将html内的文本转码存至pages？大概
        # 提取内容
        zipPath = '//a[@class="btn_b"]/@href'#zip资源的网址的位置
        zip_urlList = this_page.xpath(zipPath)  #zip资源的网址的列表，因为只有一个
        save_data(zip_urlList[0],savePath,name,'.zip')
        print(name + '_下载完成')
        howMany+=1
    print('本次已下载'+str(howMany)+'本')
    print('页数'+str(i+1)+'-'+str(pagenum))
    time.sleep(5)
print('关键词：'+key_word+'_下载完成')

