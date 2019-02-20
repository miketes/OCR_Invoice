from flask import Flask,render_template,request,Response,jsonify
import os
import math
import copy as ps
import json
from scipy import ndimage
from werkzeug.utils import secure_filename
import numpy as np
import os
import cv2 as cv
from config import *
from application import trainTicket, idcard
from apphelper.image import union_rbox,adjust_box_to_origin,base64_to_PIL
from datetime import timedelta
import Detection_tangent as detect
from Demo_tangent import *

import json
import xml.etree.ElementTree as ET
import os
import cv2
import Detection as detect_qdb
from datetime import timedelta
from ocr_main import ocr_main


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

app = Flask(__name__)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp','xml','txt'])

app.send_file_max_age_default = timedelta(seconds=1)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


if yoloTextFlag == 'keras' or AngleModelFlag == 'tf' or ocrFlag == 'torch':
    if GPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)
        import tensorflow as tf
        from keras import backend as K

        config = tf.ConfigProto()
        config.gpu_options.allocator_type = 'BFC'
        config.gpu_options.per_process_gpu_memory_fraction = 0.3  ## GPU最大占用量
        config.gpu_options.allow_growth = True  ##GPU是否可动态增加
        K.set_session(tf.Session(config=config))
        K.get_session().run(tf.global_variables_initializer())

    else:
        ##CPU启动
        os.environ["CUDA_VISIBLE_DEVICES"] = ''

if yoloTextFlag == 'opencv':
    scale, maxScale = IMGSIZE
    from text.opencv_dnn_detect import text_detect
elif yoloTextFlag == 'darknet':
    scale, maxScale = IMGSIZE
elif yoloTextFlag == 'keras':
    scale, maxScale = IMGSIZE[0], 2048
    from text.keras_detect import text_detect
elif yoloTextFlag == 'PSENET':
    scale, maxScale = 800, 1200
    from PSENET_tangent.predict_tangent import predict_image as text_detect
    from CRAFT_predict_tangent.utils_tangent import color_filter
elif yoloTextFlag == 'craft':
    scale, maxScale = 800, 1360
    from CRAFT_predict_tangent.predict_pb import predict_image as text_detect
    from CRAFT_predict_tangent.utils_tangent import color_filter

    from PSENET_tangent.predict_tangent import predict_image as text_detect_p
else:
    print("err,text engine in keras\opencv\darknet")

# from text.opencv_dnn_detect import angle_detect

if ocr_redis:
    ##多任务并发识别
    from apphelper.redisbase import redisDataBase

    ocr = redisDataBase().put_values
else:
    from crnn.keys import alphabetChinese, alphabetEnglish

    if ocrFlag == 'keras':
        from crnn.network_keras import CRNN

        if chineseModel:
            alphabet = alphabetChinese
            if LSTMFLAG:
                ocrModel = ocrModelKerasLstm
            else:
                ocrModel = ocrModelKerasDense
        else:
            ocrModel = ocrModelKerasEng
            alphabet = alphabetEnglish
            LSTMFLAG = True

    elif ocrFlag == 'torch':
        from crnn.network_torch import CRNN

        if chineseModel:
            alphabet = alphabetChinese
            if LSTMFLAG:
                ocrModel = ocrModelTorchLstm
            else:
                ocrModel = ocrModelTorchDense

        else:
            ocrModel = ocrModelTorchEng
            alphabet = alphabetEnglish
            LSTMFLAG = True
    elif ocrFlag == 'opencv':
        from run.crnn.network_dnn import CRNN

        ocrModel = ocrModelOpencv
        alphabet = alphabetChinese
    else:
        print("err,ocr engine in keras\opencv\darknet")

    nclass = len(alphabet) + 1
    if ocrFlag == 'opencv':
        crnn = CRNN(alphabet=alphabet)
    else:
        crnn = CRNN(32, 1, nclass, 256, leakyRelu=False, lstmFlag=LSTMFLAG, GPU=GPU, alphabet=alphabet)
    if os.path.exists(ocrModel):
        crnn.load_weights(ocrModel)
    else:
        print("download model or tranform model with tools!")

    ocr = crnn.predict_job





def get_time():
    from datetime import datetime
    import time
    import pytz
    return datetime.fromtimestamp(int(time.time()), pytz.timezone('Asia/Shanghai')).strftime(
        '%Y-%m-%d %H:%M:%S')

# 构建接口返回结果
def json_result(code, message, data=None, file_name=None, label_imgs=None):
    result = {
        "code": code,
        "message": message,
        "ocrIdentifyTime": get_time(),
        "data": data,
        "FileName": file_name,
        "label_imgs": label_imgs,
    }
    return jsonify(result)

@app.route('/prediction_qiandaobiao', methods=['POST','GET'])
def prediction_qiandaobiao():
    # print(request.files,'[]')

    # 校验请求参数
    if 'image' not in request.files:
        return json_result(101, "请求参数错误")
    # 获取请求参数
    file = request.files['image']
    image_name = file.filename
    # 检查文件扩展名
    if not allowed_file(image_name):
        return json_result(102, "失败，文件格式问题")
    upload_dir = "./test/"
    image_path = os.path.join(upload_dir, image_name)
    file.save(image_path)
    print ('image save as ', image_path)

    if 'xml' not in request.files:
        xml_dict = {}
        print ('No xml')
    else:
        file = request.files['xml']
        xml_name = file.filename
        xml_path = os.path.join(upload_dir, xml_name)
        file.save(xml_path)
        print('xml save as ', xml_path)

        def text2content(text):
            # 有逗号，则变为list。有小数点为float，否则为int
            contents = text.split(',')

            def int_or_float(content):
                if '.' in content:
                    return float(content)
                else:
                    return int(content)

            if len(contents) > 1:
                return [int_or_float(content) for content in contents]
            else:
                return int_or_float(contents[0])
        xml_affine_dict = {
            'scale': 'image_inference_scale',
            'pixel_filter': 'pixel_filter',
            'TEXT_PROPOSALS_MIN_SCORE': 'score_thre',
            'MAX_HORIZONTAL_GAP': 'max_dist',
            'MIN_V_OVERLAPS': 'threshold_overlap_v',
            'Adjustbox': 'move_rect',
            'scoremap_enhance_pixel': 'scoremap_enhance_pixel',
        }
        tree = ET.parse(xml_path)
        xml_content = tree.getroot()
        xml_dict = {}
        for child in xml_content[0]:
            xml_dict_key = xml_affine_dict.get(child.tag, child.tag)
            xml_dict[xml_dict_key] = text2content(child.text)

    CRAFT_params_dict = {
        'image_inference_scale': 1000,
        'pixel_filter': 10,
        'score_thre': [0.6, 0.4, 0.4],
        'max_dist': 0,
        'threshold_overlap_v': 0.5,
        'move_rect': [-5, -5, 5, 5],
        'scoremap_enhance_pixel': 2,
    }
    CRAFT_params_dict.update(xml_dict)
    if xml_dict['imgview'] == 1:
        add_img_to_result = 1
    else:
        add_img_to_result = 0

    OCR_result_dict = ocr_main(image_path, Detection_qdb, add_img_to_result=add_img_to_result, CRAFT_params_dict=CRAFT_params_dict)
    format_data = []
    for key in OCR_result_dict.keys():
        if 'table' in key:
            format_data.append(key)
            contents_dict = OCR_result_dict[key]
            for name_key in contents_dict.keys():
                content = name_key + ',' + contents_dict[name_key]['content']
                format_data.append(content)

    if xml_dict['imgview'] == 1:
        label_imgs = np.array(OCR_result_dict['label_imgs']).tolist()
    else:
        label_imgs = None

    return json_result(True, "识别成功" , data=format_data, file_name=image_name, label_imgs=label_imgs)

@app.route('/prediction_invoice', methods=['POST','GET'])
def invoice_ocr():
    # 校验请求参数
    if 'file' not in request.files:
        return fanhan_api_result(101, "请求参数错误", {}, {}, {})
    # 获取请求参数
    file = request.files['file']
    invoice_file_name = file.filename
    # 检查文件扩展名
    if not allowed_file(invoice_file_name):
        return fanhan_api_result(102, "失败，文件格式问题", {}, {}, {})
    upload_path = "./test/"
    whole_path = os.path.join(upload_path, invoice_file_name)
    file.save(whole_path)
    img = cv.imread(whole_path)

    try:
        arg = []
        tpm = []
        args = request.files['args']
        invoice_args_name = args.filename
        upload_path = "./test/"
        args_path = os.path.join(upload_path, invoice_args_name)
        args.save(args_path)

        import xml.etree.ElementTree as ET
        tree = ET.parse(args_path)
        root = tree.getroot()
        for i in root[3]:  # 遍历第一层标签
            tpm.append(i.text)

        
        for i in range(0, len(tpm)):
            if i == 0:
                tpm[i] = tpm[i].split(',')
                tpm[i] = [float(x) for x in tpm[i]]
                arg.append(tpm[i])
            elif i == 1:
                tpm[i] = tpm[i].split(',')
                tpm[i] = [float(x) for x in tpm[i]]
                arg.append(tpm[i])
            elif i == 2:
                arg.append(int(tpm[i]))
            elif i == 3:
                arg.append(float(tpm[i]))
            elif i == 4:
                tpm[i] = tpm[i].split(',')
                tpm[i] = [float(x) for x in tpm[i]]
                arg.append(tpm[i])
            elif i == 5:
                arg.append(int(tpm[i]))
            else:
                pass

    except:
        print('初始化参数')
        #
        # sc = 0
        # re = img.shape
        # if re[0] >= re[1]:
        #     if re[1] > 1200:
        #         sc = int(1200)
        #     else:
        #         sc = int(re[1] * 1.5)
        # else:
        #     if re[0] > 1200:
        #         sc = int(1200)
        #     else:
        #         sc = int(re[0] * 1.5)
        #
        # print(sc)
        arg = [0, [0.7, 0.4, 0.4], 0, 0.5, [0, 0, 0, 0], 2]
    print(arg)
    CRAFT_params_dict = {
        'pixel_filter': arg[0],
        'score_thre': arg[1],
        'max_dist': arg[2],
        'threshold_overlap_v': arg[3],
        'move_rect': arg[4],
        'scoremap_enhance_pixel': arg[5],
    }

    ocr_result,img_mat = invoice(invoice_file_name,Detection, runAngel, img_show=0, img_save=1, CRAFT_params_dict=CRAFT_params_dict)
    # for ocr_res in ocr_result:
    #     for key in ocr_res.keys():
    #         print(key, ocr_res[key])

    if len(ocr_result) >= 0:
        from datetime import datetime
        import time
        import pytz

        dic = {}
        dic['index'] = img_mat.tolist()
        dicJson = json.dumps(dic)
        ocr_identify_time = datetime.fromtimestamp(int(time.time()), pytz.timezone('Asia/Shanghai')).strftime(
            '%Y-%m-%d %H:%M:%S')
        return fanhan_api_result(True, "识别成功" , ocr_result, invoice_file_name,ocr_identify_time,label_img=dicJson)
    else:
        return fanhan_api_result(False, "识别错误", {}, {}, {})



class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, time):
            return obj.__str__()
        else:
            return super(NpEncoder, self).default(obj)

# 构建接口返回结果
def fanhan_api_result(code, message, data,file_name,ocr_identify_time,label_img=None,data_sort=None):
    result = {
        "code": code,
        "message": message,
        "data": data,
        "data_sort": data_sort,
        "FileName": file_name,
        "ocrIdentifyTime": ocr_identify_time,
        "label_img":label_img
    }
    return jsonify(result)

#排序
def find_top_box(data):
    top_id = 0
    for i, d in enumerate(data):
        if data[i]['box'][1] < data[top_id]['box'][1]:
            top_id = i
    return top_id


def find_line_box(data, id, pixel_thre):
    ids = []
    for i, d in enumerate(data):
        if abs(data[i]['box'][1] - data[id]['box'][1]) <= pixel_thre:
            ids.append(i)
    return ids


def sort_line_box(data):
    if len(data) == 1:
        return data
    flag = 1
    while flag:
        flag = 0
        data_copy = ps.deepcopy(data)
        for i, d in enumerate(data[:-1]):
            if data[i]['box'][0] > data[i + 1]['box'][0]:
                data_copy[i] = data[i + 1]
                data_copy[i + 1] = data[i]
                flag = 1
        data = ps.deepcopy(data_copy)
    return data


def find_middle_height(data):
    height_list = []
    for i, d in enumerate(data):
        height = [data[i]['box'][1], data[i]['box'][3], data[i]['box'][5], data[i]['box'][7]]
        height.sort()
        h = (height[3] + height[2] - height[1] - height[0]) // 2
        height_list.append(int(h))
    return height_list


def sort_data(data, thre):
    height_list = find_middle_height(data)
    height_list.sort()
    pixel_thre = height_list[len(height_list) // 2] * thre
    data_copy = ps.deepcopy(data)
    data_result = []
    while len(data_copy) > 0:
        top_id = find_top_box(data_copy)
        ids = find_line_box(data_copy, top_id, pixel_thre=pixel_thre)
        line_data = []
        data_remove = ps.deepcopy(data_copy)
        for id in ids:
            line_data.append(data_copy[id])
            data_remove.remove(data_copy[id])
        line_data = sort_line_box(line_data)
        data_result.append(line_data)
        data_copy = ps.deepcopy(data_remove)
    return data_result


@app.route('/prediction_img', methods=['POST','GET'])
def prediction_img():

    # 校验请求参数
    if 'file' not in request.files:
        return fanhan_api_result(101, "请求参数错误", {}, {}, {})
    # 获取请求参数
    file = request.files['file']
    invoice_file_name = file.filename
    # 检查文件扩展名
    if not allowed_file(invoice_file_name):
        return fanhan_api_result(102, "失败，文件格式问题", {}, {}, {})
    upload_path = "./test/"
    whole_path = os.path.join(upload_path, invoice_file_name)
    file.save(whole_path)
    img = cv.imread(whole_path)

    try:
        arg = []
        tpm = []
        arg2 = []
        tpm2 = []
        args = request.files['args']
        invoice_args_name = args.filename
        upload_path = "./test/"
        args_path = os.path.join(upload_path, invoice_args_name)
        args.save(args_path)

        import xml.etree.ElementTree as ET
        tree = ET.parse(args_path)
        root = tree.getroot()
        for i in root[1]:  # 遍历第一层标签
            tpm.append(i.text)

        for i in root[2]:  # 遍历每二层标签
            tpm2.append(i.text)

        for i in root[0]:  # 遍历每二层标签
            model_flg = i.text

        model_flg = int(model_flg)

        if model_flg == 1:
            model_flg = True
        else:
            model_flg = False

        for i in range(0,len(tpm)):
            if i == 0:
                arg.append(int(tpm[i]))
            elif i == 1:
                pm = int(tpm[i])
                if pm == 1:
                    arg.append(True)
                else:
                    arg.append(False)
            elif i == 2:
                arg.append(int(tpm[i]))
            elif i == 3:
                arg.append(float(tpm[i]))
            elif i == 4:
                tpm[i] = tpm[i].split(',')
                tpm[i] = [float(x) for x in tpm[i]]
                arg.append(tpm[i])
            elif i == 5:
                tpm[i] = tpm[i].split(',')
                tpm[i] = [int(x) for x in tpm[i]]
                arg.append(tpm[i])
            elif i == 6:
                arg.append(float(tpm[i]))
            elif i == 7:
                arg.append(float(tpm[i]))
            elif i == 8:
                tpm[i] = tpm[i].split(',')
                tpm[i] = [float(x) for x in tpm[i]]
                arg.append(tpm[i])
            elif i == 9:
                pm = int(tpm[i])
                if pm == 1:
                    arg.append(True)
                else:
                    arg.append(False)
            elif i == 10:
                arg.append(int(tpm[i]))
            elif i == 11:
                arg.append(float(tpm[i]))
            else:
                pass

        for i in range(0, len(tpm2)):
            if i == 0:
                arg2.append(int(tpm2[i]))
            elif i == 1:
                arg2.append(int(tpm2[i]))
            elif i == 2:
                pm = int(tpm[i])
                if pm == 1:
                    arg2.append(True)
                else:
                    arg2.append(False)
            elif i == 3:
                arg2.append(int(tpm2[i]))
            elif i == 4:
                arg2.append(float(tpm2[i]))
            elif i == 5:
                arg2.append(float(tpm2[i]))
            elif i == 6:
                arg2.append(float(tpm2[i]))
            elif i == 7:
                arg2.append(float(tpm2[i]))
            elif i == 8:
                tpm2[i] = tpm2[i].split(',')
                tpm2[i] = [float(x) for x in tpm2[i]]
                arg2.append(tpm2[i])
            else:
                pass

    except:
        print('初始化参数')

        sc = 0
        re = img.shape
        if re[0] >= re[1]:
            if re[1] > 1200:
                sc = int(1200)
            else:
                sc = int(re[1] * 1.5)
        else:
            if re[0] > 1200:
                sc = int(1200)
            else:
                sc = int(re[0] * 1.5)

        print(sc)
        arg = [sc,False,10,0.7,[0.7, 0.3, 0.3],[0, 0, 0, 0],0.,0.,[0, 0],True,3,0.5]
        arg2 = [900, 900,False, 10, 0.1, 0.6, 0.0,0.0,[0, 0, 0, 0]]
        model_flg = True


    relust ,label_img = currency_invoice(img,arg,arg2,model_flg)
    # print(relust)
    if len(relust) >= 0:
        from datetime import datetime
        import time
        import pytz

        ocr_identify_time = datetime.fromtimestamp(int(time.time()), pytz.timezone('Asia/Shanghai')).strftime(
            '%Y-%m-%d %H:%M:%S')

        #照片返回
        dic = {}
        dic['index'] = label_img.tolist()
        dicJson = json.dumps(dic)

        #返回结果排序
        sort_result = []
        data_result = sort_data(relust, thre=arg[11])


        sort_result.append(data_result)


        return fanhan_api_result(True, "识别成功" , relust, invoice_file_name,ocr_identify_time,dicJson,sort_result)
    else:
        return fanhan_api_result(False, "识别错误", {}, {}, {})


def currency_invoice(img,arg,arg2,model_flg):
    from main import TextOcrModel
    angle_detect = None
    model = TextOcrModel(ocr, text_detect, angle_detect)


    # print(arg[12],'---------------------',type(arg[12]))
    if model_flg == True:
        model = TextOcrModel(ocr, text_detect, angle_detect)
        rotate_img = color_filter(img, color_thre=255, mode='less')
        img = color_filter(rotate_img, color_thre=50, mode='more')
        if lab == '-1':
            result, angle, im_show = model.model_CRAFT(img,
                                                   scale= arg[0],
                                                   detectAngle= False,  ##是否进行文字方向检测
                                                   MAX_HORIZONTAL_GAP= arg[2],  ##字符之间的最大间隔    #30
                                                   MIN_V_OVERLAPS= arg[3],        #0.5
                                                   TEXT_PROPOSALS_MIN_SCORE= arg[4],  #[0.5, 0.2, 0.7],
                                                   leftAdjustAlph= arg[6],  ##
                                                   rightAdjustAlph= arg[7],  ##
                                                   Adjustbox= arg[5],  ##
                                                   pixel_filter= arg[8],  ##
                                                   batch_by_1= arg[9],
                                                   scoremap_enhance_pixel = arg[10],
                                                       )
        elif lab == '0':
            result, angle, im_show = model.model_CRAFT(img,
                                                   scale= arg[0],
                                                   detectAngle= False,  ##是否进行文字方向检测
                                                   MAX_HORIZONTAL_GAP= arg[2],  ##字符之间的最大间隔    #30
                                                   MIN_V_OVERLAPS= arg[3],        #0.5
                                                   TEXT_PROPOSALS_MIN_SCORE= arg[4],  #[0.5, 0.2, 0.7],
                                                   leftAdjustAlph= arg[6],  ##
                                                   rightAdjustAlph= arg[7],  ##
                                                   Adjustbox= arg[5],  ##
                                                   pixel_filter= arg[8],  ## 参数一是过滤宽，第二个参数是过滤高
                                                   batch_by_1= arg[9],
                                                   scoremap_enhance_pixel = arg[10],
                                                       )


        # ti2 = time.asctime(time.localtime(time.time()))
        # print("本地时间为 :", ti2)

    else:
        model = TextOcrModel(ocr, text_detect_p, angle_detect)
        rotate_img = color_filter(img, color_thre=255, mode='less')
        img = color_filter(rotate_img, color_thre=50, mode='more')

        # ------------------------变换---------------------------
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(gray, 50, 150, apertureSize=3)

        # 霍夫变换
        lines = cv.HoughLines(edges, 1, np.pi / 180, 0)

        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

        # print(x1, '---------', x2, '---------', x2, '---------', y2)
        if x1 == x2 or y1 == y2 or y2 == -1000:
            rotate_img = img

        else:
            t = float(y2 - y1) / (x2 - x1)
            rotate_angle = math.degrees(math.atan(t))
            if rotate_angle > 45:
                rotate_angle = -90 + rotate_angle
            elif rotate_angle < -45:
                rotate_angle = 90 + rotate_angle

            rotate_img = ndimage.rotate(img, rotate_angle)
        result, angle, im_show = model.model_PSENET(rotate_img,
                                                    scale=arg2[0],
                                                    maxScale=arg2[1],
                                                    detectAngle=arg2[2],  ##是否进行文字方向检测
                                                    MAX_HORIZONTAL_GAP=arg2[3],  ##字符之间的最大间隔
                                                    MIN_V_OVERLAPS=arg2[4],
                                                    TEXT_PROPOSALS_MIN_SCORE=arg2[5],
                                                    leftAdjustAlph=arg2[6],  ##
                                                    rightAdjustAlph=arg2[7],  ##
                                                    Adjustbox=arg2[8],  ##
                                                    )

    # result = union_rbox(result, 0.2)
    res = [{'text': x['text'],
            'name': str(i),
            'box': {'cx': x['cx'],
                    'cy': x['cy'],
                    'w': x['w'],
                    'h': x['h'],
                    'angle': x['degree']

                    }
            } for i, x in enumerate(result)]
    res = adjust_box_to_origin(img, angle, res)  ##修正box

    return res,im_show

@app.route('/train_tickets', methods=['POST','GET'])
def train_tickets():
    # 校验请求参数
    if 'file' not in request.files:
        return fanhan_api_result(101, "请求参数错误", {}, {}, {})

    # 获取请求参数
    file = request.files['file']
    invoice_file_name = file.filename

    # 检查文件扩展名
    if not allowed_file(invoice_file_name):
        return fanhan_api_result(102, "失败，文件格式问题", {}, {}, {})

    upload_path = "./invoice-api/run/test/"
    whole_path = os.path.join(upload_path, invoice_file_name)
    file.save(whole_path)

    relust = train_ticket(whole_path)
    # print(relust)
    if len(relust) >= 0:
        from datetime import datetime
        import time
        import pytz

        ocr_identify_time = datetime.fromtimestamp(int(time.time()), pytz.timezone('Asia/Shanghai')).strftime(
            '%Y-%m-%d %H:%M:%S')
        return fanhan_api_result(True, "识别成功", relust, invoice_file_name, ocr_identify_time)
    else:
        return fanhan_api_result(False, "识别错误", {}, {}, {})

def train_ticket(filepath):
    scale, maxScale = IMGSIZE[0], 2048
    from run.text.keras_detect import text_detect
    from run.main import TextOcrModel
    angle_detect = None
    model = TextOcrModel(ocr, text_detect,angle_detect)

    img = cv.imread(filepath)
    result, angle = model.model(img,
                                scale=scale,
                                maxScale=maxScale,
                                detectAngle = False,  ##是否进行文字方向检测
                                MAX_HORIZONTAL_GAP=100,  ##字符之间的最大间隔
                                MIN_V_OVERLAPS=0.6,
                                MIN_SIZE_SIM=0.6,
                                TEXT_PROPOSALS_MIN_SCORE=0.1,
                                TEXT_PROPOSALS_NMS_THRESH=0.3,
                                TEXT_LINE_NMS_THRESH=0.99,  ##iou值
                                LINE_MIN_SCORE=0.1,
                                leftAdjustAlph=0.01,  ##
                                rightAdjustAlph=0.01,  ##
                                )

    res = trainTicket.trainTicket(result)
    res = res.res
    res = [{'text': res[key], 'name': key, 'box': {}} for key in res]

    return res


if __name__ == '__main__':
    # threaded=False, processes=3
    runAngel = False
    path_model = r"./models/invoice_VAT.pb"
    path_craft_model = r"./models/weight.pb"
    path_direct_model = r""
    if (runAngel == True):
        path_direct_model = r"Angle-model.pb"
    else:
        path_direct_model = None

    path_pbtxt = r"./labels_pbtxt/invoiceDetection.pbtxt"  # 载入标签文件

    NUM_CLASSES = 35  # 分类数量
    Detection = detect.invoiceDetection(path_model, path_craft_model, path_pbtxt, NUM_CLASSES,
                                        path_direct_model)  # 构建预测类

    path_craft_model = r".、weight.pb"
    Detection_qdb = detect_qdb.Detection_craft(path_craft_model)

    app.run(host='0.0.0.0', port=7799, threaded=True)