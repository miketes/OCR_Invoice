import requests
import json
import base64
# import run.data.sql as sql
from flask import jsonify

# 首先将图片读入

# 由于要发送json，所以需要对byte进行str解码

def getByte(path):
    with open(path, 'rb') as f:
        img_byte = base64.b64encode(f.read())

    img_str = img_byte.decode('ascii')

    return img_str

# 构建接口返回结果

img_str = getByte('./test/IMG.jpg')
url = 'http://127.0.0.1:7799/invoice-ocr'          #currency_invoice      train_tickets/火车票        invoice-ocr

files = {'file': ('1.jpg', open(r'./img', 'rb'))}
res = requests.post(url, files=files)

data = res.text
# print(data)
result = json.loads(data)


for i in result.values():
    print(i)
#
# sql.insrt_into_invoice(result['data'],result['FileName'])
# da = build_api_result(True, "识别成功" , res, '1.jpg',data)
# print(da)