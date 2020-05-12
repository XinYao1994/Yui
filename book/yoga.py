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
        sleep(1)
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

    def start_book(self):
        self.driver = Browser(driver_name=self.driver_name, executable_path=self.executable_path)
        #窗口大小的操作
        self.driver.driver.set_window_size(2800, 2000)
        while True:
            # sleep(1.5)
            # find today
            # find two days later
            # not login, login
            self.login()
            # self.driver.visit(self.url)
            # get all available date

            # filter
            # self.driver.select("filter-2", 707) # filter the teacher
            # self.driver.select("filter-4", 4) # filter the teacher
            # self.driver.find_by_id('filter-2').first.click()
            # self.driver.find_by_id('filter-4').first.click()
            # print("1111111111111111111111111111111")
            # print(self.driver.find_by_css('span[class=selection]'))# .first.click()
            # print(len(self.driver.find_by_css('span[class=selection]')))# .first.click()
            # for i in self.driver.find_by_css('span[class=selection]'):
            #     print(i)
            # print("1111111111111111111111111111111")
            sels= self.driver.find_by_css('span[class=selection]')
            # teacher = "Andrew Petker\r\n"
            sels[3].click()
            # sels[3].find_by_css('span[class=select2-selection__rendered]').first.fill(teacher)
            # self.driver.find_by_css('input[class=select2-search__field]').first.fill(teacher)
            self.driver.find_by_text('Andrew Petker').first.click()
            sleep(1)
            sels[5].click()
            self.driver.find_by_text('程度2').first.click()
            sleep(1)
            # sels[3].click()
            # sels[5].click()
            # sels[5].click()
            # self.driver.find_by_id('filter-2').first.select(707)
            # self.driver.find_by_id('filter-4').first.select(4)
            yoga_class = self.driver.find_by_text('候补名单') # .click()
            yoga_class[1].click()
            exit()
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

