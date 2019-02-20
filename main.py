#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main 
@author: xxx
"""
from text.detector.detectors import TextDetector
from apphelper.image import rotate_cut_img, sort_box
import numpy as np
import cv2 as cv
from PIL import Image


class TextOcrModel(object):
    def __init__(self, ocrModel, textModel, angleModel):
        self.ocrModel = ocrModel
        self.textModel = textModel
        self.angleModel = angleModel

    def detect_angle(self, img):
        """
        detect text angle in [0,90,180,270]
        @@img:np.array
        """
        angle = self.angleModel(img)
        if angle == 90:
            im = Image.fromarray(img).transpose(Image.ROTATE_90)
            img = np.array(im)
        elif angle == 180:
            im = Image.fromarray(img).transpose(Image.ROTATE_180)
            img = np.array(im)
        elif angle == 270:
            im = Image.fromarray(img).transpose(Image.ROTATE_270)
            img = np.array(im)

        return img, angle

    def detect_box(self, img, scale=600, maxScale=900):
        """
        detect text angle in [0,90,180,270]
        @@img:np.array
        """
        boxes, scores = self.textModel(img, scale, maxScale)
        return boxes, scores

    def box_cluster(self, img, boxes, scores, **args):

        MAX_HORIZONTAL_GAP = args.get('MAX_HORIZONTAL_GAP', 100)
        MIN_V_OVERLAPS = args.get('MIN_V_OVERLAPS', 0.6)
        MIN_SIZE_SIM = args.get('MIN_SIZE_SIM', 0.6)
        textdetector = TextDetector(MAX_HORIZONTAL_GAP, MIN_V_OVERLAPS, MIN_SIZE_SIM)

        shape = img.shape[:2]
        TEXT_PROPOSALS_MIN_SCORE = args.get('TEXT_PROPOSALS_MIN_SCORE', 0.7)
        TEXT_PROPOSALS_NMS_THRESH = args.get('TEXT_PROPOSALS_NMS_THRESH', 0.3)
        TEXT_LINE_NMS_THRESH = args.get('TEXT_LINE_NMS_THRESH', 0.3)
        LINE_MIN_SCORE = args.get('LINE_MIN_SCORE', 0.8)

        boxes, scores = textdetector.detect(boxes,
                                            scores[:, np.newaxis],
                                            shape,
                                            TEXT_PROPOSALS_MIN_SCORE,
                                            TEXT_PROPOSALS_NMS_THRESH,
                                            TEXT_LINE_NMS_THRESH,
                                            LINE_MIN_SCORE
                                            )
        return boxes, scores

    def ocr_batch(self, img, boxes, leftAdjustAlph=0.0, rightAdjustAlph=0.0):
        """
        batch for ocr
        """
        im = Image.fromarray(img)

        newBoxes = []
        for index, box in enumerate(boxes):
            partImg, box = rotate_cut_img(im, box, leftAdjustAlph, rightAdjustAlph)

            img = np.array(partImg)
            _, img_bright = cv.threshold(img, 200, 255, cv.THRESH_BINARY)
            # cv.imshow('mg0',img_bright)
            # cv.waitKey()
            box['img'] = partImg.convert('L')

            newBoxes.append(box)

        res = self.ocrModel(newBoxes)
        return res

    def model(self, img, **args):
        detectAngle = args.get('detectAngle', False)
        if detectAngle:
            img, angle = self.detect_angle(img)
        else:
            angle = 0
        scale = args.get('scale', 608)
        maxScale = args.get('maxScale', 608)
        boxes, scores = self.detect_box(img, scale, maxScale)  ##文字检测
        boxes, scores = self.box_cluster(img, boxes, scores, **args)
        boxes = sort_box(boxes)
        leftAdjustAlph = args.get('leftAdjustAlph', 0)
        rightAdjustAlph = args.get('rightAdjustAlph', 0)

        res = self.ocr_batch(img, boxes, leftAdjustAlph, rightAdjustAlph)

        return res, angle

    def model_PSENET(self, img, **args):
        detectAngle = args.get('detectAngle', False)
        if detectAngle:
            img, angle = self.detect_angle(img)
        else:
            angle = 0
        scale = args.get('scale', 608)
        maxScale = args.get('maxScale', 608)
        MAX_HORIZONTAL_GAP = args.get('MAX_HORIZONTAL_GAP', 15)
        MIN_V_OVERLAPS = args.get('MIN_V_OVERLAPS', 0.01)
        TEXT_PROPOSALS_MIN_SCORE = args.get('TEXT_PROPOSALS_MIN_SCORE', 0.9)
        Adjustbox = args.get('Adjustbox', [-5, -5, 5, 5])
        print('scale', scale, MAX_HORIZONTAL_GAP)
        boxes = \
            self.textModel(img, min_len=scale, max_len=maxScale, score_thre=TEXT_PROPOSALS_MIN_SCORE,
                           max_dist=MAX_HORIZONTAL_GAP, threshold_overlap_v=MIN_V_OVERLAPS, move_rect=Adjustbox)
        import cv2
        im = img.copy()
        bo = []
        for rt in boxes:
            if not (rt[0] == rt[2] or rt[1] == rt[3]):
                bo.append(rt)
            cv2.rectangle(im, (rt[0], rt[1]), (rt[2], rt[3]), (0, 0, 255), 2)

        boxes = np.array(bo)
        im_show = cv2.resize(im, (im.shape[1] // 2, im.shape[0] // 2))

        image = cv2.cvtColor(im_show, cv2.COLOR_BGR2RGB)
        # cv2.imshow('img2', im_show)
        # cv2.waitKey(0)

        boxes_8 = np.zeros((len(boxes), 8), np.int32)
        for i in range(len(boxes)):
            boxes_8[i, 0] = boxes[i, 0]  # x1
            boxes_8[i, 1] = boxes[i, 1]  # y1
            boxes_8[i, 2] = boxes[i, 2]  # x2
            boxes_8[i, 3] = boxes[i, 1]  # y1
            boxes_8[i, 4] = boxes[i, 2]  # x2
            boxes_8[i, 5] = boxes[i, 3]  # y2
            boxes_8[i, 6] = boxes[i, 0]  # x1
            boxes_8[i, 7] = boxes[i, 3]  # y2

        # print(boxes_8,'boxes_8')
        boxes = sort_box(boxes_8)

        # print(boxes,'boxes')
        leftAdjustAlph = args.get('leftAdjustAlph', 0)
        rightAdjustAlph = args.get('rightAdjustAlph', 0)
        res = self.ocr_batch(img, boxes, leftAdjustAlph, rightAdjustAlph)
        return res, angle, image

    def model_CRAFT(self, img, **args):
        detectAngle = args.get('detectAngle', False)
        if detectAngle:
            img, angle = self.detect_angle(img)
        else:
            angle = 0
        scale = args['scale']
        MAX_HORIZONTAL_GAP = args['MAX_HORIZONTAL_GAP']
        MIN_V_OVERLAPS = args['MIN_V_OVERLAPS']
        TEXT_PROPOSALS_MIN_SCORE = args['TEXT_PROPOSALS_MIN_SCORE']
        Adjustbox = args['Adjustbox']
        pixel_filter = args['pixel_filter']
        batch_by_1 = args['batch_by_1']
        scoremap_enhance_pixel = args['scoremap_enhance_pixel']

        boxes = \
            self.textModel(img, image_inference_scale=scale, score_thre=TEXT_PROPOSALS_MIN_SCORE, batch_by_1=batch_by_1,
                           max_dist=MAX_HORIZONTAL_GAP, threshold_overlap_v=MIN_V_OVERLAPS, move_rect=Adjustbox,
                           pixel_filter=pixel_filter,scoremap_enhance_pixel=scoremap_enhance_pixel)

        bo = []
        for rt in boxes:
            Xs = [rt[0], rt[2], rt[4], rt[6]]
            Ys = [rt[1], rt[3], rt[5], rt[7]]

            x1 = min(Xs)
            x2 = max(Xs)
            y1 = min(Ys)
            y2 = max(Ys)
            hight = y2 - y1
            width = x2 - x1
            # print(hight, '-------------------', width)

            if hight <= 10 or hight >= 300:
                pass
            else:
                bo.append(rt)


        boxes = sort_box(bo)
        leftAdjustAlph = args.get('leftAdjustAlph', 0)
        rightAdjustAlph = args.get('rightAdjustAlph', 0)

        # for rt in boxes:
            # cv.rectangle(img, (rt[0], rt[1]), (rt[2], rt[3]), (0, 0, 255), 2)

        res = self.ocr_batch(img, boxes, leftAdjustAlph, rightAdjustAlph)

        import cv2
        im = img.copy()

        for pts in boxes:
            # print(pts)
            pts = pts.reshape((-1, 1, 2))
            # print()
            # print(pts)
            cv2.polylines(im, [pts], True, (0, 0, 255), 1)

            # cv2.imshow('img',im)
            # cv2.waitKey()

        # cv.imwrite('label.jpg',im)
        return res, angle, im

