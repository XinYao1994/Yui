import os
import sys
from functools import reduce
import requests
from bs4 import BeautifulSoup
import ddddocr

parent_dir = os.path.dirname(os.path.abspath(__file__))
ebook_download_dir = parent_dir + "/ebooks/"

class Ebook:
    def __init__(self) -> None:
        self.url = ""
        self.ocr = ddddocr.DdddOcr(beta=True)
        self.ban_list = ["mobi", "azw3"]
    
    def search_book(self):
        pass
    
    def download_book(self):
        pass

    def is_banned(self, name):
        for type_ in self.ban_list:
            if type_ in name:
                return True
        return False

class Ebook2(Ebook):
    def __init__(self) -> None:
        super().__init__()
        self.url = "https://ebook2.lorefree.com"

    def search_book(self, title):
        web_data = requests.get(self.url + "/site/index?s=" + title)
        soup = BeautifulSoup(web_data.content, "html.parser")
        select_resource = soup.select('.book-content')

        def get_herf(res):
            return res.find('a', href = True).attrs['href']
            
        book_url = list(map(get_herf, select_resource))
        return book_url
    
    def download_book(self, book_url):
        for burl in book_url:
            book_data = requests.get(self.url + burl)
            soup = BeautifulSoup(book_data.content, "html.parser")
            select_resource = soup.select('.book-file')

            captchCode = soup.find("img", {"id":"captchaimg"}).attrs['src']
            # with open("tmp.png", "wb") as f:
            #     f.write(image.content)

            def get_book_info(res):
                bookname = res.findAll("span", {"class":"label-default"})[1].text
                fileid = res.find("input", {"name":"fileid"}).attrs['value']
                bookid = res.find("input", {"name":"bookid"}).attrs['value']
                return (bookname, bookid, fileid)

            book_info = list(map(get_book_info, select_resource))
            all_ban = reduce(lambda y,x: self.is_banned(x[0]) and y, book_info, True)
            print(book_info)
            if all_ban:
                continue

            image = requests.get(self.url + captchCode)
            captchCode = self.ocr.classification(image.content)
            print(captchCode)

            for binfo in book_info:
                if self.is_banned(binfo[0]):
                        continue
                # /book/down?bookid='+bookId+'&fileid='+fileId+'&captchCode='+captchCode;
                fin_url = "/book/down?bookid="+binfo[1]+"&fileid="+binfo[2]+"&captchCode="+captchCode
                # print(fin_url)
                fin_data = requests.get(self.url + fin_url)
                with open(ebook_download_dir+binfo[0], "wb") as f:
                    f.write(fin_data.content)
                print("done")
                return True
        return False

class Zh_1lib(Ebook):
    def __init__(self) -> None:
        super().__init__()
        self.url = "https://zh.1lib.net"

    def search_book(self, title):
        web_data = requests.get(self.url + "/s/" + title)
        soup = BeautifulSoup(web_data.content, "html.parser")
        select_resource = soup.select('.z-book-precover')

        def get_herf(res):
            return res.find('a', href = True).attrs['href']
        
        book_url = list(map(get_herf, select_resource))
        # print(book_url)
        return book_url

    def download_book(self, book_url):
        for burl in book_url:
            book_data = requests.get(self.url + burl)
            soup = BeautifulSoup(book_data.content, "html.parser")
            select_resource = soup.select('.details-buttons-container')
            # print(select_resource) 
            def get_herf(res):
                return res.find('a', href = True).attrs['href']
            
            book_info = list(map(get_herf, select_resource))  
            # print(book_info)
            for binfo in book_info:
                fin_data = requests.get(self.url + binfo)
                with open(ebook_download_dir+binfo[0], "wb") as f:
                    f.write(fin_data.content)
                print("done")
                return True
        return False

ebook_url = [
    Ebook2(),
    Zh_1lib()
]

if __name__ == '__main__':
    book_titles = []

    if len(sys.argv) > 2:
        # read data from file
        book_titles.extend(sys.argv[1:])
    if len(book_titles) == 0:
        # read from file
        with open(ebook_download_dir + "title.txt", "r", encoding="utf-8") as f: # "gbk"
            title_data = f.readlines()
            title_data = list(map(lambda x: x.replace("\n", ""), title_data))
            title_data = list(map(lambda x: x.replace("\xa0", ""), title_data))
            book_titles.extend(title_data)

    if len(book_titles) == 0:
        print("No specified book")
        exit()
    
    for title in book_titles:
        if len(title) == 0:
            continue
        print(title)
        ret = 0
        for web in ebook_url:
            book_url = web.search_book(title)
            ret = web.download_book(book_url)
            if ret: break
