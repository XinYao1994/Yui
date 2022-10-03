
# scrapy startproject [projectName]

# define item
'''
import scrapy

class Day1Item(scrapy.Item):
    city = scrapy.Field()
    temperature = scrapy.Field()
    date = scrapy.Field() 
    pass
'''

# spiders
'''
import scrapy
from day1.items import Day1Item    # day1是文件夹的名，Day1Item是items.py中的类class名
class weatherSpider(scrapy.spiders.Spider):   #weatherSpider是自定义的名
    name = "sina"       #sina是自定义的名
    allowed_domains = ['sina.com.cn']   #sina.com.cn是限定在这个网站的范围之内爬虫
    start_urls = ['http://weather.sina.com.cn/xian']  #开始爬虫的网址
    def parse(self, response):
        item= Day1Item()
        item['city'] = response.xpath('//*[@class="slider_ct_name"]/text()').extract()
        item['temperature']=response.xpath('//*[@class="wt_fc_c0_i_temp"]/text()').extract()
        item['date']=response.xpath('//*[@class="wt_fc_c0_i_date"]/text()').extract()
        return item
'''

# 
'''
BOT_NAME = 'day1'  
 
SPIDER_MODULES = ['day1.spiders']
NEWSPIDER_MODULE = 'day1.spiders'
 
FEED_EXPORT_ENCODING = 'utf-8'    
'''
