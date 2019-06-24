import thulac	

thu1 = thulac.thulac()
'''
sentence = [
    "我爱北京天安门",
    "今天的北京天气不错",
]

text = thu1.cut(sentence[0], text=True)  
text = thu1.cut(sentence[1], text=True) 

ERNIE model for chinese
'''
data = open("./write/data/chinese.txt", encoding='utf-8').read()

sentence = data.split("\n")

for x in sentence:
    if x=="":
        continue
    text = thu1.cut(x, text=True)
    print(text)



