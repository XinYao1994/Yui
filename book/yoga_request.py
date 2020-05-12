import requests
from bs4 import BeautifulSoup
import json
# from jsonpath import jsonpath

class Booking(object):
    def __init__(self):
        self.login = self.login_yoga
        #
    def config(self, url, username, passwd):
        self.url = url
        self.username = username
        self.passwd = passwd

    def login_yoga(self):
        self.login_url = "https://pure360-api.pure-international.com/api/v3/login"

        userAgent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36"
        header = {
            # "accept": "application/json, text/javascript, */*; q=0.01", 
            # "accept-encoding": "gzip, deflate, br", 
            # "accept-language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7", 
            # "content-type": "application/json; charset=UTF-8",
            "origin": "https://pure360.pure-yoga.com",
            "referer": self.url,
            # "sec-fetch-dest": "empty",
            # "sec-fetch-mode": "cors",
            # "sec-fetch-site": "cross-site",
            "user-agent": userAgent,
            "x-date": self.x_date,
            "x-token": self.x_token,
        }
        payload = {
            "jwt": True,
            "language_id": 3,
            "password": "gy123456",
            "platform": "Web",
            "username": "yuguo1128@gmail.com",
            "region_id": 1,
        }
        print(header)
        print(payload)
        response = self.sess.post(self.login_url, data=payload, headers = header)
        # print(f"statusCode = {response.status_code}")
        # print(f"text = {response.text}")
        # print(f"statusCode = {response.data}")
        # print(response.content)

    def access(self, url):
        userAgent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36"
        header = {
            "Connection": "keep-alive",
            "Host": "pure360.pure-yoga.com",
            "Referer": self.url,
            "User-Agent": userAgent,
            'Connection':"keep-alive",
        }
        # 
        response = self.sess.get(url, headers = header)
        print(f"statusCode = {response.status_code}")
        # print(f"text = {response.text}")
        scripts = BeautifulSoup(response.text).find_all('script')
        for sc in scripts:
            if "jQueryHeaders" in str(sc):
                format_str = str(sc)
        tokens = format_str.split("\n")[1]
        tokens = tokens[tokens.find('{'): tokens.find('}')+1]
        tokens_json = json.loads(tokens)
        print(tokens_json)
        self.x_date = str(tokens_json["X-Date"])
        self.x_token = str(tokens_json["X-Token"])
        print(self.x_date)
        print(self.x_token)

    def start_book(self):
        self.sess = requests.Session()
        self.access(self.url) # 
        self.login_yoga()


if __name__ == '__main__':
    # URL
    yoga_url = "https://pure360.pure-yoga.com/zh-cn/HK?location_id=11"
	# 用户名
    username = 'yuguo1128@gmail.com'
    # 密码
    password = 'gy123456'

    robot = Booking()
    robot.config(yoga_url, username, password)
    robot.start_book() # 从今天开始订

