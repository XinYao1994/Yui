import sys
import os
import importlib
importlib.reload(sys)

# pip3 install pdfminer3k
# ref :
# https://www.cnblogs.com/wj-1314/p/9429816.html
# https://lishuaishuai.iteye.com/blog/2436919
# https://zhuanlan.zhihu.com/p/33945312
# http://denis.papathanasiou.org/archive/2010.08.04.post.pdf
# https://euske.github.io/pdfminer/programming.html

from pdfminer.pdfparser import PDFParser,PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LTTextBoxHorizontal, LAParams, LTTextBox, LTTextLine, LTChar, LTFigure, LTImage, LTLine, LTRect, LTCurve
from pdfminer.pdfinterp import PDFTextExtractionNotAllowed

# https://github.com/dpapathanasiou/pdfminer-layout-scanner
def write_file (folder, filename, filedata, flags='w'):
    """Write the file data to the folder and filename combination
    (flags: 'w' for write text, 'wb' for write binary, use 'a' instead of 'w' for append)"""
    result = False
    if os.path.isdir(folder):
        try:
            file_obj = open(os.path.join(folder, filename), flags)
            file_obj.write(filedata)
            file_obj.close()
            result = True
        except IOError:
            pass
    return result

# https://github.com/dpapathanasiou/pdfminer-layout-scanner
from binascii import b2a_hex
def determine_image_type (stream_first_4_bytes):
    """Find out the image file type based on the magic number comparison of the first 4 (or 2) bytes"""
    file_type = None
    bytes_as_hex = b2a_hex(stream_first_4_bytes)
    if bytes_as_hex.startswith('ffd8'):
        file_type = '.jpeg'
    elif bytes_as_hex == '89504e47':
        file_type = '.png'
    elif bytes_as_hex == '47494638':
        file_type = '.gif'
    elif bytes_as_hex.startswith('424d'):
        file_type = '.bmp'
    return file_type


images_folder = "./data/img/"
def PDF_parse(path):
    fp = open(path, 'rb')
    _pdf_praser = PDFParser(fp)
    _pdf = PDFDocument()
    _pdf_praser.set_document(_pdf)
    _pdf.set_parser(_pdf_praser)

    #_pdf.initialize("password")
    _pdf.initialize()
    content = []

    if not _pdf.is_extractable:
        # use py ocr to analysis
        return
    else:
        _pdf_manager = PDFResourceManager()
        _pdf_la = LAParams()
        _pdf_devive = PDFPageAggregator(_pdf_manager, laparams=_pdf_la)
        _pdf_ip = PDFPageInterpreter(_pdf_manager, _pdf_devive)

        for _page in _pdf.get_pages():
            _pdf_ip.process_page(_page)
            _layout = _pdf_devive.get_result()
            # LTPage LTTextBox LTTextLine LTChar/LTAnno LTFigure LTImage LTLine LTRect LTCurve
            #for e in _layout:
            content.append(objs_parser(_layout))

    return content

def objs_parser(objs, times=0, text_content=None):
    if times > 3:
        return []
    if text_content == None:
        text_content = []
    for e in objs: 
        if(isinstance(e, LTTextBoxHorizontal)):
            #print("text")
            ret = e.get_text()
            text_content.append(ret)
            # #print(ret)
        if isinstance(e, LTImage):  # 图片对象
            print("image")
            if e.stream:
                _image = e.stream.get_rawdata()
                if _image:
                    print(_image[0:4])
                    file_ext = determine_image_type(_image[0:4])
                    print(file_ext)
                    file_name = ''.join(['_', e.name, file_ext])
                    if file_ext:
                        write_file(images_folder, file_name, _image, flags='wb')
                    text_content.append(file_name)
        if isinstance(e, LTFigure):  # figure对象
            print("figure")
            print(times)
            #text_content.append(objs_parser(e, times+1, text_content))
        #if isinstance(e, LTCurve):  # 曲线对象
        #    do nothing
    return ''.join(text_content).replace('\n', '')

if __name__ == '__main__':
    _pdf_content = PDF_parse("C:/Users/Xin/Downloads/1802.06952.pdf")
    print(_pdf_content)



