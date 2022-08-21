# -*- coding:utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('gbk')
import os 
import time  
import webbrowser
from PIL import Image
import pytesseract

PATH = lambda p: os.path.abspath(p)  



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
	ret = pytesseract.image_to_string(Image.open(path + "/" + "4.png"), lang='chi_sim')
	result = ""
	for i in ret.split('\n'):
		result += i
		if i.find('?') != -1:
			break
		if i.find('7') != -1:
			break
	print(result)
	webbrowser.open('https://www.baidu.com/s?wd='+result)
	os.popen("adb shell rm /data/local/tmp/tmp.png") 
	print("success")

if __name__ == "__main__": 
	#print "打印中文字符".decode('utf-8').encode('gb18030')
	while True:
		TrySearch()
		p = raw_input()




