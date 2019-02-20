import requests
import json

def http_post(body_u=None):

  print()
  print(body_u)
  print()
  url = 'http://ocr.mypiaoge.cn/piaoge/api/dd/log'

  body = {'userName': "T10000001", 'password': "25A1F51E-F643-4F07-910D-75FDF28C912F"}
  headers = {'content-type': "application/json"}

  # body_u = {
  #   'FPDM': "1100183130",
  #   'FPHM': "11671464",
  #   'FPRQ': "20190621",
  #   'FPJE': "10.15",  # 专票必填
  #   'FPLX': "01",  # 专票=01  普票=04
  #   'taxNO': ""
  #
  # }
  response = requests.post(url, data=json.dumps(body), headers=headers)

  jso = response.text
  js = json.loads(jso)
  jss = js['state']
  # print(js)

  if jss == True:
    jsw = js['data']
    token = (jsw['token'])

    # print('--------------------------------')
    ocr = 'http://ocr.mypiaoge.cn/piaoge/api/InvVerify/invInfo?token=' + token
    responses = requests.post(ocr, data=json.dumps(body_u), headers=headers)

    joq = responses.text
    jo = json.loads(joq)
    gh = jo['state']
    # ga = jo['sta']
    print(gh)
    jo = jo['data']
    # print(jo)
    print()
    print()
  else:
    jo = None
    gh = 'False'
    # print(gh)
  return jo,gh


# http_post()