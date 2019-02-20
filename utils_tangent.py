import numpy as np
import cv2
import os
import time
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
import math
import skimage
import copy
import re
import matplotlib.pyplot as plt

#########################################################################
# boxes_utils
#########################################################################

def point2point_distance(pointPx, pointPy):
    # 点到点的距离
    distance = ((pointPx[0] - pointPy[0])**2 + (pointPx[1] - pointPy[1])**2)**0.5
    return distance

def point2line_distance(x1, y1, x2, y2, pointPx, pointPy):
    # 求点到直线的距离
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - y1 * x2
    distance = abs(A * pointPx + B * pointPy + C) / ((A * A + B * B) ** 0.5)
    return distance

def box_scale(box, scale):
    h = (box[2] - box[0])*scale
    w = (box[3] - box[1])*scale
    center_h = (box[2] + box[0])/2
    center_w = (box[3] + box[1])/2
    return int(center_h-h/2), int(center_w-w/2), int(center_h+h/2), int(center_w+w/2)

def box_pixel_scale(box, pixel):
    # 按照像素对box进行放大缩小
    return box[0]-pixel, box[1]-pixel, box[2]+pixel, box[3]+pixel

def bbox_four2eight(bbox):
    # bbox 4点转为8点
    return np.array([[bbox[0], bbox[1]], [bbox[0], bbox[3]], [bbox[2], bbox[3]], [bbox[2], bbox[1]]])

def cal_boxes_centers(boxes):
    boxes = np.array(boxes)
    box_centers = []
    for box in boxes:
        box_center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
        box_centers.append(box_center)
    box_centers = np.array(box_centers)
    box_center = np.mean(box_centers, axis=0)
    return box_center

def union_box(box1, box2):
    u_box = np.zeros(4, np.float32)
    u_box[0] = np.maximum(box1[0], box2[0])
    u_box[1] = np.maximum(box1[1], box2[1])
    u_box[2] = np.minimum(box1[2], box2[2])
    u_box[3] = np.minimum(box1[3], box2[3])
    if u_box[2]>u_box[0] and u_box[3]>u_box[1]:
        return u_box
    else:
        return None

def cal_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0

def cal_self_iou(box1, box2):
    # box1与box2的交集面积 / box1的面积
    u_box = union_box(box1, box2)
    if u_box is None:
        return 0
    u_area = (u_box[2]-u_box[0]) * (u_box[3]-u_box[1])
    box1_area = (box1[2]-box1[0]) * (box1[3]-box1[1])
    return u_area/box1_area

def combine_boxes(boxes):
    # boxes合并
    boxes = np.array(boxes)
    boxes_combine = [np.min(boxes[:, 0]), np.min(boxes[:, 1]), np.max(boxes[:, 2]), np.max(boxes[:, 3])]
    return np.array(boxes_combine)

def sort_boxes(boxes, height_endurance_scale, mode='xy'):
    # 对boxes进行排序后输出
    assert mode in ['xy', 'yx']
    if mode is 'xy':
        k = 1
    else:
        k = 0
    boxes = boxes[np.argsort(boxes[:, k])]
    boxes_h = (boxes[:, k + 2] - boxes[:, k])
    boxes_h.sort()
    height_endurance = (boxes_h[len(boxes) // 2]) * height_endurance_scale

    boxes_sorted = []
    boxes_num = len(boxes)
    boxes_ids = list(range(boxes_num))
    for i in range(boxes_num):
        if i not in boxes_ids:
            continue
        line_ids = [i]
        boxes_ids.remove(i)
        for id in boxes_ids.copy():
            if id not in boxes_ids:
                continue
            if abs(boxes[i, k + 2] + boxes[i, k] - boxes[id, k + 2] - boxes[id, k]) / 2 <= height_endurance:
                line_ids.append(id)
                boxes_ids.remove(id)
        boxes_line = boxes[line_ids]
        boxes_line = boxes_line[np.argsort(boxes_line[:, 1 if k == 0 else 0])]
        boxes_sorted.append(boxes_line)
    return boxes_sorted

def filter_double(Boxes, Scores, Classes):
    # 根据score对box进行筛选，用于YOLO
    c_count = Counter(Classes)
    c_doubles = [key for key,value in c_count.items()if value > 1]
    if len(c_doubles)>0:
        remove_list = []
        for c_double in c_doubles:
            remove_ids = np.where(Classes==c_double)[0].tolist()
            s_double = np.where(Classes==c_double, Scores, 0.)
            s_max_id = np.argmax(s_double)
            remove_ids.remove(s_max_id)
            remove_list.extend(remove_ids)
        Boxes, Scores, Classes = map(lambda boxes:np.delete(boxes, remove_list, 0), (Boxes, Scores, Classes))
    return Boxes, Scores, Classes

def cal_bbox_area(bbox):
    return (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])

def check_bbox1_in_bbox2(bbox1, bbox2):
    if bbox1[0] > bbox2[0] and bbox1[1] > bbox2[1] and bbox1[2] < bbox2[2] and bbox1[3] < bbox2[3]:
        return True
    return False

def check_bbox1_in_bbox2_by_bbox1center(bbox1, bbox2):
    center1_h = (bbox1[0] + bbox1[2]) / 2
    center1_w = (bbox1[1] + bbox1[3]) / 2
    if center1_h < bbox2[2] and center1_h > bbox2[0] and center1_w < bbox2[3] and center1_w > bbox2[1]:
        return True
    return False

def find_areamax_bbox(bboxes):
    area = 0
    result = bboxes[0]
    for bbox in bboxes:
        if cal_bbox_area(bbox) > area:
            area = cal_bbox_area(bbox)
            result = bbox
    return result
#########################################################################
#  Visualize utils
#########################################################################

def color_dec2hex(color_dec):
    color_hex = '#'
    for c in color_dec:
        hex_c = hex(c).replace('0x', '')
        if len(hex_c)==1:
            hex_c = '0'+hex_c
        color_hex += hex_c
    return color_hex

def draw_rec(image, boxes, texts=None, color=(255,0,0), img_show=False, img_save=None):
    image_copy = image.copy()
    h, w = image.shape[:2]
    for i, box in enumerate(boxes):
        if np.max(boxes) <= 1.0:
            draw_box1 = int(box[1]*w), int(box[0]*h)
            draw_box2 = int(box[3]*w), int(box[2]*h)
        else:
            draw_box1 = int(box[1]), int(box[0])
            draw_box2 = int(box[3]), int(box[2])
        cv2.rectangle(image_copy, draw_box1, draw_box2, color, 2)
        if texts:
            text = texts[i]
            pilimg = Image.fromarray(image_copy)
            draw = ImageDraw.Draw(pilimg)
            size = 30
            font = ImageFont.truetype("./font/simhei.ttf", size)
            color_hex = color_dec2hex(color)
            #draw.text((draw_box1[0], draw_box1[1]-size), text, fill=color_hex, font=font)
            draw.text((draw_box1[0], draw_box1[1]), text, fill=color_hex, font=font)
            image_copy = np.array(pilimg)
    if img_show:
        im = Image.fromarray(image_copy)
        im.show()
    elif img_save:
        im = Image.fromarray(image_copy)
        im.save(img_save)
    return image_copy

def draw_area_dict_on_image(image, contents_dict, img_show=False, img_save=None):
    image_show = image.copy()
    for key in contents_dict.keys():
        box = contents_dict[key]['box']
        text = contents_dict[key]['content']
        image_show = draw_rec(image_show, [box], [text], (255,0,0))
    if img_show:
        im = Image.fromarray(image_show)
        im.show()
    if img_save:
        im = Image.fromarray(image_show)
        im.save(img_save)
        print('image save as: ', img_save)
    return image_show

def show_bbox(image, bbox, img_show=1):
    image_show = image.copy()
    if np.max(bbox)<=1:
        h, w = image.shape[:2]
        bbox = [int(bbox[0]*h), int(bbox[1]*w), int(bbox[2]*h), int(bbox[3]*w)]

    image_show[bbox[0]:bbox[2],bbox[1]:bbox[3],0] = 255
    image_show[bbox[0]:bbox[2],bbox[1]:bbox[3],1] = 0
    image_show[bbox[0]:bbox[2],bbox[1]:bbox[3],2] = 0
    if img_show:
        fig = plt.figure(figsize=(16,16))
        plt.imshow(image_show)
    else:
        return image_show[bbox[0]:bbox[2],bbox[1]:bbox[3]], image_show

#########################################################################
#   utils
#########################################################################


#########################################################################
#  表格处理工具
#########################################################################

def filter_frame(image, bg_color, line_thickness = 6):
    new_h = 96
    h, w = image.shape[:2]
    new_w = int(w / h * new_h)
    image = cv2.resize(image, (new_w, new_h))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)  # 50,150,3
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 5, minLineLength=30, maxLineGap=5)  # 650,50,20

    image_copy = image.copy()
    if isinstance(lines, np.ndarray):
        for line in lines:
            x1, y1, x2, y2 = line[0]
            theta = math.atan(float(y2 - y1) / float(x2 - x1 + 1e-4))
            theta = abs(theta / np.pi * 180)
            endurance = 5
            horizontal_condition = theta < endurance and theta > -endurance
            vertical_condition = theta < 90 + endurance and theta > 90 - endurance
            if horizontal_condition or vertical_condition:
                cv2.line(image_copy, (x1, y1), (x2, y2), bg_color, line_thickness)
    return image_copy

def edges_canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)  # 50,150,3
    return edges

def image_rotate_correction(image, theta_endurance):
    rows, cols, channels = image.shape
    image_copy = image.copy()

    ##### 旋转校正Rotation #####
    # 统计图中长横线的斜率来判断整体需要旋转矫正的角度
    edges = edges_canny(image)
    # Image.fromarray(edges).show()
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 500, 0, minLineLength=50, maxLineGap=50)  # 650,50,20
    pi = np.pi
    theta_list = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        theta = math.atan(float(y2 - y1) / float(x2 - x1 + 0.001))
        theta_deg = theta / pi * 180
        if theta_deg < theta_endurance and theta_deg > -theta_endurance:
            theta_list.append(theta_deg)
            cv2.line(image_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
    assert len(theta_list) > 0, '旋转校正没有找到合适的角度，扫描件有问题？'
    theta_average = np.mean(theta_list)
    affineShrinkTranslationRotation = cv2.getRotationMatrix2D((0, rows), theta_average, 1)
    ShrinkTranslationRotation = cv2.warpAffine(image, affineShrinkTranslationRotation, (cols, rows))
    # image_copy = cv2.warpAffine(image_copy, affineShrinkTranslationRotation, (cols, rows))
    # Image.fromarray(image_copy).show()
    return ShrinkTranslationRotation

def pure_image_by_mean_color(image, expansion_ratio=0):
    color_thre = np.mean(image)
    h, w = image.shape
    h_start = 0
    h_end = h
    for i in range(h):
        if np.mean(image[i, :]) <= color_thre:
            h_start = i
            break
    for i in range(h):
        if np.mean(image[h - 1 - i, :]) <= color_thre:
            h_end = h - 1 - i
            break
    if h_start >= h_end:
        h_start = 0
        h_end = h

    w_start = 0
    w_end = w
    for i in range(w):
        if np.mean(image[:, i]) <= color_thre:
            w_start = i
            break
    for i in range(w):
        if np.mean(image[:, w - 1 - i]) <= color_thre:
            w_end = w - 1 - i
            break
    if w_start >= w_end:
        w_start = 0
        w_end = w

    if expansion_ratio:
        size = h_end - h_start
        h_start -= int(size * expansion_ratio)
        h_end += int(size * expansion_ratio)
        w_start -= int(size * expansion_ratio)
        w_end += int(size * expansion_ratio)
    image_crop = image[h_start:h_end, w_start:w_end]
    return image_crop, (h_start, w_start, h_end, w_end)

def cal_bg_color_by_smallerthanmean(image):
    bg_color = [None, None, None]
    for i in range(3):
        image_channel = image[:, :, i]
        color_mean = np.mean(image_channel)
        small_ids = np.where(image_channel>color_mean)
        bg_color[i] = int(np.mean(image_channel[small_ids]))
    return tuple(bg_color)

def find_char_id(text, chartext):
    ids = []
    for id, t in enumerate(text):
        if t in chartext:
            ids.append(id)
    return ids

def number_correct(text):
    key_words_dict = {
        'oO': '0',
        '/liI()[]L': '1',
        'rzZ': '2',
        # '': '3',
        'f': '4',
        'sS': '5',
        'b': '6',
        # '': '7',
        # '': '8',
        'pP': '9',
    }
    for key in key_words_dict.keys():
        for w in key:
            text = text.replace(w, key_words_dict[key])
    text = ''.join(re.findall('\d+', text))
    return text

def scale_boxes_by_center(bbox, scale):
    # 以box的中心点，按scale进行缩放
    # bbox: [y1,x1,y2,x2]
    h = bbox[2] - bbox[0]
    w = bbox[3] - bbox[1]
    center_y = bbox[0] + h / 2
    center_x = bbox[1] + w / 2
    return int(center_y-scale*h/2), int(center_x-scale*w/2), int(center_y+scale*h/2), int(center_x+scale*w/2)

def coordinate_relative2absolute(rel, abs):
    # rel: 相对坐标， 0~1
    # abs: 绝对坐标， int
    h = abs[2] - abs[0]
    w = abs[3] - abs[1]
    y1 = abs[0] + h*rel[0]
    x1 = abs[1] + w*rel[1]
    y2 = abs[0] + h*rel[2]
    x2 = abs[1] + w*rel[3]
    return y1, x1, y2, x2

def check_key_words_num(text, key_words):
    cnt = 0
    for t in text:
        cnt += 1 if t in key_words else 0
    return cnt

def filter_dummy(area_result):
    # 'boxes', 'texts', 'imgs' 版本
    remove_ids = []
    texts = area_result['texts']
    for i, text in enumerate(texts):
        if len(text) == 0:
            remove_ids.append(i)

    area_result_copy = copy.deepcopy(area_result)
    cnt = 0
    for remove_id in remove_ids:
        for key in ['boxes', 'texts', 'imgs']:
            remove_num = remove_id-cnt
            if isinstance(area_result[key], np.ndarray):
                area_result_copy[key] = np.delete(area_result[key], remove_num, 0)
            else:
                del (area_result_copy[key][remove_num])
        cnt += 1

    return area_result_copy

def get_frame(image, h_thre, s_thre, v_thre):
    # 阈值参考：https://blog.csdn.net/Lily_9/article/details/83114633
    image_HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    h_cond = np.logical_and(image_HSV[: ,: ,0] >= h_thre[0], image_HSV[: ,: ,0] <= h_thre[1])
    s_cond = np.logical_and(image_HSV[: ,: ,1] >= s_thre[0], image_HSV[: ,: ,1] <= s_thre[1])
    v_cond = np.logical_and(image_HSV[: ,: ,2] >= v_thre[0], image_HSV[: ,: ,2] <= v_thre[1])
    where_cond = np.logical_and(np.logical_and(h_cond, s_cond), v_cond)
    where_cond = np.tile(where_cond[: ,: ,np.newaxis], (1 ,1 ,3))
    image_frame = np.where(where_cond, image, 255)
    return image_frame


