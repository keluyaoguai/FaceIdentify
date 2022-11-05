import time
import requests
import re
from lxml import etree
import os
import urllib.parse
import unicodedata
#//*选取文档中的所有元素
#//*[@what]选取文档中的所有元素中拥有what属性的部分
#控制搜索内容
key_word = input("请输入小说关键词\n")
#是否允许关键词在作者名中
#writer_if = int(input('是否允许作者名存在关键字|1/0\n'))
writer_if = 0
#控制下载页数
start_page = int(input("请输入开始页数\n"))
#控制开始位置
start_ordinal=int(input("请输入开始序位\n"))
keyWord_url= urllib.parse.quote(key_word)
novel_namePath = '//*[@class="book_textList2 bd"]/li/a/text()'  # 该关键词搜索内容的小说名
writer_namePath = '//*[@class="book_textList2 bd"]/li/text()'
pages_numPath = '//*[@class="pagination clearfix"]/span/text()'#该关键词搜索内容的总页数
pages_numPath = '//div[@class="pagination clearfix"]/span/text()'#该关键词搜索内容的总页数
searchResult = 'http://downnovel.com/search.htm?keyword='+keyWord_url+'&pn=1'#初始网页
#searchResult = 'http://downnovel.com/search.htm?keyword=%E6%96%97%E7%BD%97&pn=1'#初始网页
zipPath = '//a[@class="btn_b"]/@href'#zip资源的网址
theUrl = 'http://downnovel.com'
rootPath = 'E:/spider_file/novel/'
savePath = rootPath+key_word+'/'
howMany =0


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
def save_data(url,savePath,nname,wname,type):
    '''资源网址，保存的文件夹路径，文件名称，文件类型(前面有点)'''
    data = get_data(url)  # data为byte字节
    f = open(savePath+nname+'_'+wname+type, mode='wb')
    f.write(data)
    f.close()
def download(nname,wname,url,itsPage,itsOrdinal):
    wname = wname.replace('?','_')
    global howMany
    thisNovel = theUrl + url  # 小说网址
    # print(thisNovel)
    # 下载网页数据
    thisPage = requests.get(thisNovel)  # 该对象储存着该网页的所有内容
    this_page = etree.HTML(thisPage.text)  # 将html内的文本转码存至pages？大概
    # 提取内容
    zipPath = '//a[@class="btn_b"]/@href'  # zip资源的网址的位置
    zip_urlList = this_page.xpath(zipPath)  # zip资源的网址的列表，因为只有一个
    save_data(zip_urlList[0], savePath,nname,wname,'.zip')
    logs = nname+'_'+wname +'_下载完成' + '^' + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())+key_word+'_'+str(itsPage)+'_'+str(itsOrdinal))
    print(logs)
    logw(logs)
    howMany += 1
    time.sleep(5)


#获取总页数
check = requests.get(searchResult)
firstPage = etree.HTML(check.text)
pagesNum = firstPage.xpath(pages_numPath)  # 这里是一页内小说的网址的列表
pagenum = int(pagesNum[0].split(r"/")[1])


def logw(logs):
    log = open(rootPath+'log.txt', mode='a')
    log.write(logs+'\n')
    log.close()
for i in range(start_page,pagenum+1):#爬全部页数的
#for i in range(1):#爬指定页数的
    next_searchResult = re.sub('&pn=\d+', '&pn=%d' % i, searchResult)
    #下载网页数据
    thePage = requests.get(next_searchResult)#该对象储存着该网页的所有内容
    the_page = etree.HTML(thePage.text)  # 将html内的文本转码存至pages？大概
    #提取内容
    novel_urlPath = '//*[@class="book_textList2 bd"]/li/a/@href'  # 该关键词搜索内容的novel_url
        #xpath方法生成的变量确实是list格式，可以用list的方法操作
    novel_nameList = the_page.xpath(novel_namePath)#这里是一页内小说的名字的列表
    writer_nameList = the_page.xpath(writer_namePath)#这里是一页内作者名字的列表
    novel_urlList = the_page.xpath(novel_urlPath)#这里是一页内小说的网址的列表

    print('开始下载页数' + str(i) + '-' + str(pagenum))
    itsOrdinal = 1#在第一页可能不准
    for nname, wname,url in zip(novel_nameList,writer_nameList, novel_urlList):
        unicodedata.normalize('NFKC', wname)
        wname = wname[3:-1]
        if start_ordinal ==1:
            if writer_if ==1:
                download(nname,wname,url,i,itsOrdinal)
            elif writer_if == 0:
                if key_word in nname:
                    download(nname,wname,url,i,itsOrdinal)
                else:
                    pass
        elif start_ordinal>1:
            start_ordinal -=1
        itsOrdinal+=1
    print('本次已下载'+str(howMany)+'本')
    logw('已完成<'+key_word+'>'+str(i)+'-'+str(pagenum)+'页')
    time.sleep(5)
print('关键词：'+key_word+'_下载完成')


#添加筛选关键词是否在作者名中的功能|完成（原本是检查作者名中是否存在，但实际上根据需求，检查书名中是否存在才是最合适的）
#添加重复书名但作者不同的功能（通过文件名加作者名实现）|完成
#添加日志功能，记录已下载小说的书名及下载时间以及完成页数|完成
#添加超时重启功能(检测时间，检测当前位置，重启)|
#删除作者名中的非法词汇|（？）完成

