# -*- coding:utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('gbk')
import os 
import time  
from aip import AipOcr
import webbrowser

PATH = lambda p: os.path.abspath(p)  
APP_ID = '10698090'
API_KEY = 'Npu9CMGunnG7gPYogu9Qzo2H'
SECRET_KEY = 'uOGDaHmGGzGFn9AQbnhxZk9ngqOdE4yI'

client = AipOcr(APP_ID, API_KEY, SECRET_KEY)

def get_file_content(filePath):
	with open(filePath, 'rb') as fp:
		return fp.read()

def TrySearch():  
	path = PATH(os.getcwd() + "/screenshot")  
	timestamp = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))  
	#os.popen("adb wait-for-device")  
	os.popen("adb shell screencap -p /data/local/tmp/tmp.png")  
	if not os.path.isdir(PATH(os.getcwd() + "/screenshot")):  
		os.makedirs(path)  
	os.popen("adb pull /data/local/tmp/tmp.png " + PATH(path + "/" + timestamp + ".png"))
	image = get_file_content(PATH(path + "/" + timestamp + ".png"))
	#image = get_file_content(PATH(os.getcwd() + "/screenshot/4.png"))
	#options = {}
	#options["language_type"] = "CHN_ENG"
	#options["detect_direction"] = "true"
	#options["detect_language"] = "true"
	#options["probability"] = "true"
	ret = client.basicGeneral(image);#, options
	result = ""
	#reslist = []
	for i in ret['words_result']:
		result += i['words']
		if i['words'].find('?') != -1:
			break;
		#reslist.append(i['words'])
	#print result
	#print(repr(reslist).decode('unicode-escape'))
	webbrowser.open('https://www.baidu.com/s?wd='+result)
	os.popen("adb shell rm /data/local/tmp/tmp.png") 
	print "success"  

if __name__ == "__main__": 
	#print "打印中文字符".decode('utf-8').encode('gb18030')
	while True:
		TrySearch()
		p = raw_input();




