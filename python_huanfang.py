import os
import re

def lookup(path):
    names = os.listdir(path)
    ret = []
    for strname in names:
        if re.match("test*[0-9]*[0-9]*[0-9]*[0-9]*.log", strname, flags=0) != None:
            ret.append(strname)
        if re.match("test*[0-9]*[0-9]*[0-9]*[0-9]*[0-9]*.log", strname, flags=0) != None:
            ret.append(strname)
        if re.match("test*[0-9]*[0-9]*[0-9]*[0-9]*[0-9]*[0-9]*.log", strname, flags=0) != None:
            ret.append(strname)
        if re.match("test*[0-9]*[0-9]*[0-9]*[0-9]*[0-9]*[0-9]*[0-9]*.log", strname, flags=0) != None:
            ret.append(strname)
        if re.match("test*[0-9]*[0-9]*[0-9]*[0-9]*[0-9]*[0-9]*[0-9]*[0-9]*.log", strname, flags=0) != None:
            ret.append(strname)
    return ret

print(lookup("D:\WorkSpace"))

# 1. 对于LR, 请给出赋予每个赝本不同的权重的L2正则化逻辑回归的
# 交叉熵函数以及参数更新的过程





