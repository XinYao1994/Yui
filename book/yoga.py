from splinter.browser import Browser
from time import sleep
import traceback

class Booking(object):
    def __init__(self):
        self.driver_name = "chrome"
        self.executable_path = r"C:\Users\Xin\AppData\Local\Programs\Python\Python36\Scripts\chromedriver.exe"
    
    def config(self, url, username, passwd):
        self.url = url
        self.username = username
        self.passwd = passwd
        self.login = self.login_yoga

    def login_yoga(self):
        self.driver.visit(self.url)
        sleep(0.5)
        try:
            self.driver.find_by_text('登入').click()
            sleep(1)
            # cli
            # self.driver.fill('username', self.username) #login-modal.sign-in-form.
            # 
            # self.driver.fill('password', self.passwd) #login-modal.sign-in-form.
            # self.driver.find_by_id('sign-in-form').first.find_by_id('username').first.fill(self.username)
            # self.driver.find_by_id('sign-in-form').first.find_by_id('password').first.fill(self.passwd)
            self.driver.find_by_id('username').first.fill(self.username)
            self.driver.find_by_id('password').first.fill(self.passwd)
            #sleep(1)
            self.driver.find_by_name('login').first.click()
        except Exception:
            pass
    
    def book(self):
        yoga_class_table = self.driver.find_by_id('schedule-list')
        print(len(yoga_class_table))
        # 
        yoga_class = yoga_class_table[0].find_by_xpath("//tr[@data-time=\"20:30\"]")
        print(len(yoga_class))
        yoga_days = yoga_class[0].find_by_tag("td")
        print(len(yoga_days))
        yoga_book = yoga_days[4].find_by_tag("button")
        print(len(yoga_book))
        try:
            yoga_book[0].click()
        except Exception:
            yoga_class = yoga_class_table[0].find_by_xpath("//tr[@data-time=\"20:30\"]")
            print(len(yoga_class))
            yoga_days = yoga_class[0].find_by_tag("td")
            print(len(yoga_days))
            yoga_book = yoga_days[5].find_by_tag("button")
            print(len(yoga_book))
            yoga_book[0].click()
            yoga_book[1].click()


    def start_book(self):
        self.driver = Browser(driver_name=self.driver_name, executable_path=self.executable_path)
        #窗口大小的操作
        self.driver.driver.set_window_size(2800, 2000)
        while True:
            sleep(0.5)
            self.login()
            # find today
            # find two days later
            # not login, login
            # self.driver.visit(self.url)
            # get all available date
            '''
            # filter
            sels= self.driver.find_by_css('span[class=selection]')
            sels[3].click()
            self.driver.find_by_text('Andrew Petker').first.click()
            sleep(1)
            sels[5].click()
            self.driver.find_by_text('程度1').first.click()
            sleep(1)
            # sels[3].click()
            # sels[5].click()
            # sels[5].click()
            # self.driver.find_by_id('filter-2').first.select(707)
            # self.driver.find_by_id('filter-4').first.select(4)
            '''
            # book
            try:
                self.book()
            except Exception:
                pass
            # yoga_class = self.driver.find_by_text('候补名单') # .click()
            # yoga_class[1].click()
            # exit()
            # book noon and afternoon


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

