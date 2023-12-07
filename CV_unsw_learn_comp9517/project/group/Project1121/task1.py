#!/usr/bin/env python
# COMP9517 2020 T3
# Project Task1 by Group VGR, Haojin, Guo
# # Tue Nov 10

import cv2
import random
import numpy as np
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
from skimage import feature
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split


def get_imgs_n_csvs(path, name, num):
    imgs_n_csv = []
    for i in reversed(range(1, num + 1)):
        temp = []
        rgb_name = name + str(i).rjust(2, '0') + '_rgb.png'
        bbox_name = name + str(i).rjust(2, '0') + '_bbox.csv'
        img_path = os.path.join(path, rgb_name)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        temp.append(img_rgb)
        # read test+csv
        bbox_path = os.path.join(path, bbox_name)
        img_csv = pd.read_csv(bbox_path, header=None)
        temp.append(img_csv)
        imgs_n_csv.append(temp)
    return imgs_n_csv


def get_pos_neg_samples(imgs_n_csv):
    pos_imgs = []
    neg_imgs = []
    for img_n_csv in imgs_n_csv:
        img = img_n_csv[0]
        icsv = img_n_csv[1]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        r, c, l = img.shape
        # draw bounding box and get the e_image inside of the box
        for j in range(len(icsv)):
            x1, y1, x2, y2, x3, y3, x4, y4 = icsv.loc[j, :]
            eimg = img_gray[y1:y3, x1:x3]
            eimg_res = cv2.resize(eimg, (200, 200), interpolation=cv2.INTER_CUBIC)
            pos_imgs.append(eimg_res)
            # cut some negative samples randomly
            for i in range(4):
                xm = x1 + random.randint(-150, 150)
                ym = y1 + random.randint(-150, 150)
                if (xm != x1) and (ym != y1) and (0 < xm < c) and (0 < ym < r):
                    xmw = xm + random.randint(17, 217)
                    ymh = ym + random.randint(17, 217)
                    dx = min(x3, xmw) - max(x1, xm)
                    dy = min(y3, ymh) - max(y1, ym)
                    ovl_rate = dx * dy / ((y3 - y1) * (x3 - x1))
                    if (xmw != x3) and (ymh != y3) and (0 < xmw < c) and (0 < ymh < r) and (ovl_rate <= 0.136):
                        neg_eimg = img_gray[ym:ymh, xm:xmw]
                        neg_eimg_res = cv2.resize(neg_eimg, (200, 200), interpolation=cv2.INTER_CUBIC)
                        neg_imgs.append(neg_eimg_res)
    return pos_imgs, neg_imgs


def get_hog_feature_n_labels(pos_train_eimgs, neg_train_eimgs):
    # Get hog feature vector, get positive and negative sample label value
    x_train_eimgs_feat = []
    y_train_labels = []
    for eimg in pos_train_eimgs:
        eimg_hog = feature.hog(eimg, feature_vector=True)
        x_train_eimgs_feat.append(eimg_hog)
        y_train_labels.append(1)
    for neg_eimg in neg_train_eimgs:
        eimg_hog_neg = feature.hog(neg_eimg, feature_vector=True)
        x_train_eimgs_feat.append(eimg_hog_neg)
        y_train_labels.append(0)
    return x_train_eimgs_feat, y_train_labels


# detection based on hsv
def HSV_based_box(img):
    dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    # ，min（35，43，46）max(77, 255, 255)
    # greenLower = np.array([31, 40, 44])
    greenLower = np.array([34, 43, 46])
    greenUpper = np.array([77, 255, 255])
    # rgb-->hsv
    hsv = cv2.cvtColor(dst, cv2.COLOR_RGB2HSV)
    # mask: Remove colors other than green
    mask = cv2.inRange(hsv, greenLower, greenUpper)

    # Add the mask and the image pixel by pixel, bit operation, and extract the green part
    green = cv2.bitwise_and(hsv, hsv, mask=mask)
    img_temp = cv2.cvtColor(green, cv2.COLOR_HSV2BGR)
    img_hsv_2_gray = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_hsv_2_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    img_dila = cv2.dilate(thresh, kernel, thresh, iterations=6)
    # cv2.RETR_EXTERNAL,,,,Only the contour profile is detected
    _, contours, hierarchy = cv2.findContours(img_dila, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #  return x,y,w,h
    boxes = [cv2.boundingRect(c) for c in contours]
    return boxes


def sifting_boxes(boxes):  # detect small boxes that overlap in the  same area
    center_points = []
    sift_boxes = []
    for box in boxes:
        x, y, w, h = box
        point = np.array([x + w / 2, y + h / 2])
        detect_fator = False
        for i in range(len(center_points)):
            distance = np.sqrt(np.sum(np.square(point - center_points[i])))
            if 0 <= distance <= 100:
                detect_fator = True
                # get new local center point
                x1 = (point[0] + center_points[i][0]) / 2
                y1 = (point[1] + center_points[i][1]) / 2
                #  get sifting bounding boxes
                sift_boxes[i][0] = min(x, sift_boxes[i][0])
                sift_boxes[i][1] = min(y, sift_boxes[i][1])
                sift_boxes[i][2] = max(w, sift_boxes[i][2])
                sift_boxes[i][3] = max(h, sift_boxes[i][3])
        if detect_fator == False:
            center_points.append(np.array(point))
            sift_boxes.append([x, y, w, h])
    return sift_boxes


def get_csv_boxes(img_csv):
    csv_bounding_boxes = []
    for j in range(len(img_csv)):
        x1, y1, x2, y2, x3, y3, x4, y4 = img_csv.loc[j, :]
        csv_bounding_boxes.append((x1, y1, x3, y3))
    return csv_bounding_boxes


# get the  new bounding boxes after SVM classifier filtering
def get_new_boxes_1(boxes, img_gray, clf):
    new_bounding_boxes = []
    for box in boxes:
        x, y, w, h = box
        e_img = img_gray[y:(y + h), x:(x + w)]
        e_img_res = cv2.resize(e_img, (200, 200), interpolation=cv2.INTER_CUBIC)
        e_img_hog = hog(e_img_res, feature_vector=True)
        e_img_hog = e_img_hog.reshape(1, -1)
        pred_eimg = clf.predict(e_img_hog)
        if pred_eimg == 1:
            score = clf.decision_function(e_img_hog)
            new_bounding_boxes.append((x, y, w, h, score))
    return new_bounding_boxes


# detect the predict box is true or not, and counts
def precision_by_img_count(new_boxes, csv_boxes):
    count = 0
    for index, new_box in enumerate(new_boxes):
        x, y, w, h, score = new_box
        p1 = np.array([x + w / 2, y + h / 2])
        area_1 = w * h
        for csv_box in csv_boxes:
            x1, y1, x3, y3 = csv_box
            area_2 = (x3 - x1) * (y3 - y1)
            p2 = np.array([(x3 + x1) / 2, (y3 + y1) / 2])
            distance = np.sqrt(np.sum(np.square(p1 - p2)))
            area_ratio = min(area_1, area_2) / max(area_1, area_2)
            if distance < 100 and area_ratio > 0.45:
                count += 1
    precision = 0
    if len(new_boxes) >= len(csv_boxes):
        if count >= len(csv_boxes):
            count = len(csv_boxes)

    precision_recall = count / len(new_boxes)
    precision_preci = count / len(csv_boxes)
    #     else:
    #         precision_recall = count / len(new_boxes)
    #         precision_preci = count / len(csv_boxes)
    #
    # elif len(new_boxes) < len(csv_boxes):
    #     precision_recall = count / len(csv_boxes)
    precision = (precision_recall + precision_preci) / 2
    return precision, count


def precision_by_bounding_area(new_boxes, csv_boxes):
    precis = []
    new_boxes = sorted(new_boxes, key=lambda new_boxes: new_boxes[4], reverse=True)
    if (len(new_boxes) > len(csv_boxes)) or (len(new_boxes) == len(csv_boxes)):
        new_boxes_rj = new_boxes[:len(csv_boxes)]
        for index, new_box in enumerate(new_boxes_rj):
            x, y, w, h, score = new_box
            p1 = np.array([x + w / 2, y + h / 2])
            area_1 = w * h
            xw = x + w
            yh = y + h
            for csv_box in csv_boxes:
                x1, y1, x3, y3 = csv_box
                area_2 = (x3 - x1) * (y3 - y1)
                p2 = np.array([(x3 + x1) / 2, (y3 + y1) / 2])
                distance = np.sqrt(np.sum(np.square(p1 - p2)))
                area_ratio = min(area_1, area_2) / max(area_1, area_2)
                if (distance < 100) and (area_ratio > 0.45):
                    # overlap area
                    dx = min(x3, xw) - max(x1, x)
                    dy = min(y3, yh) - max(y1, y)
                    if (dx >= 0) and (dy >= 0):
                        preci = (dx * dy) / area_2
                        precis.append(preci)
    if len(new_boxes) < len(csv_boxes):
        differ = len(csv_boxes) - len(new_boxes)
        for i in range(differ):
            precis.append(0)
        for index, csv_box in enumerate(csv_boxes):
            x1, y1, x3, y3 = csv_box
            area_2 = (x3 - x1) * (y3 - y1)
            p2 = np.array([(x3 + x1) / 2, (y3 + y1) / 2])
            for new_box in new_boxes:
                x, y, w, h, score = new_box
                p1 = np.array([x + w / 2, y + h / 2])
                area_1 = w * h
                xw = x + w
                yh = y + h
                distance = np.sqrt(np.sum(np.square(p1 - p2)))
                area_ratio = min(area_1, area_2) / max(area_1, area_2)
                if (distance < 100) and (area_ratio > 0.45):
                    dx = min(x3, xw) - max(x1, x)
                    dy = min(y3, yh) - max(y1, y)
                    if (dx >= 0) and (dy >= 0):
                        preci = (dx * dy) / area_2
                        precis.append(preci)
    if len(precis) > 0:
        if len(precis) < len(csv_boxes):
            for i in range(len(csv_boxes) - len(precis)):
                precis.append(0)
        if len(precis) >= len(csv_boxes):
            precis = precis[:len(csv_boxes)]
        one_img_ap = np.mean(precis)
    else:
        one_img_ap = 0
    return one_img_ap


def draw_contrast_bounding(new_boxes, csv_boxes, img):
    # mark the prediction box as orange
    for nbox in new_boxes:
        x, y, w, h, score = nbox
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 191, 0), 25)
    # mark the true box as blue
    for cbox in csv_boxes:
        x1, y1, x3, y3 = cbox
        cv2.rectangle(img, (x1, y1), (x3, y3), (0, 191, 255), 12)
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def compute_AP_element(test_images_csv, clf):
    prec_by_count = []
    prec_by_area = []
    for i in range(len(test_images_csv)):
        image_n_csv = test_images_csv[i]
        # for image_n_csv in test_images_csv:
        img = image_n_csv[0]
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_csv = image_n_csv[1]
        csv_boxes = get_csv_boxes(img_csv)
        boxes = HSV_based_box(img)
        sift_boxes = sifting_boxes(boxes)
        # print("Merging the local boxes completed: %s" % time.ctime())
        new_boxes = get_new_boxes_1(sift_boxes, img_gray, clf)

        # draw two contrast bounding boxes  on rgb image
        draw_contrast_bounding(new_boxes, csv_boxes, img)
        print("Drawing the contrast bounding boxes out of the image: %s" % time.ctime())
        # calculate the precision of each image
        # method 1 by the number of image;
        preci_1, count = precision_by_img_count(new_boxes, csv_boxes)
        prec_by_count.append(preci_1)
        print(f"The total number of predicted plants in image-{i} is {count}.")
        # method 2 by each area of box
        preci_2 = precision_by_bounding_area(new_boxes, csv_boxes)
        prec_by_area.append(preci_2)
    return prec_by_count, prec_by_area


def main():
    print("Start : %s\n" % time.ctime())
    tray_path = r'../Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Tray'
    Ara2012_path_1 = os.path.join(tray_path, 'Ara2012')
    Ara2013_Canon_path_2 = os.path.join(tray_path, 'Ara2013-Canon')
    Ara2013_RPi_path_3 = os.path.join(tray_path, 'Ara2013-RPi')

    imgs_n_csv_1 = get_imgs_n_csvs(path=Ara2012_path_1, name='ara2012_tray', num=16)
    imgs_n_csv_2 = get_imgs_n_csvs(path=Ara2013_Canon_path_2, name='ara2013_tray', num=27)
    imgs_n_csv_3 = get_imgs_n_csvs(path=Ara2013_RPi_path_3, name='ara2013_tray', num=27)

    train_images_csv_1, test_images_csv_1 = train_test_split(imgs_n_csv_1, train_size=0.8, test_size=0.2)
    train_images_csv_2, test_images_csv_2 = train_test_split(imgs_n_csv_2, train_size=0.8, test_size=0.2)
    train_images_csv_3, test_images_csv_3 = train_test_split(imgs_n_csv_3, train_size=0.8, test_size=0.2)

    train_images_csv = train_images_csv_1 + train_images_csv_2 + train_images_csv_3
    test_images_csv = test_images_csv_1 + test_images_csv_2 + test_images_csv_3

    print("The number of train samples: ", len(train_images_csv))
    print("The number of test samples: ", len(test_images_csv))
    print()
    # get positive  and negative samples
    pos_train_eimgs, neg_train_eimgs = get_pos_neg_samples(train_images_csv)

    # training a  SVM classifer
    feat_train, train_labels = get_hog_feature_n_labels(pos_train_eimgs, neg_train_eimgs)

    # clf = SVC(kernel="linear")
    clf = LinearSVC()
    clf.fit(feat_train, train_labels)
    print("Complete the SVM classifier : %s\n" % time.ctime())

    # get two kind of precision list
    prec_by_count, prec_by_area = compute_AP_element(test_images_csv, clf)

    # print(prec_by_count)
    # print(prec_by_area)

    AP_img = np.mean(prec_by_count)
    AP_area = np.mean(prec_by_area)
    print()
    print("Average precision based on correct bounding boxes counts:  %.4f" % AP_img)
    print("Average precision based on correct boundary areas:  %.4f" % AP_area)
    print()
    print("End : %s" % time.ctime())


if __name__ == "__main__":
    main()

# from sklearn.cluster import KMeans
# from collections import Counter
# import cv2 as cv
# import numpy as np
# from matplotlib import pyplot as plt
#
#
# def a_b_tfb(img):
#     i = img[:, :, 1]
#     j = img[:, :, 0]
#     f = cv.blur(i, ksize=(3, 3)) + (cv.GaussianBlur(j, (3, 3), 4) - cv.GaussianBlur(j, (3, 3), 1))
#     tfb = np.exp(-np.abs(f) / 50)
#     return np.array([i, j, tfb]).transpose(1, 2, 0), tfb
#
#
# def DIC_IOU(seg, gt):
#     its = np.sum(seg[gt == 255])
#     seg_gt = (np.sum(seg) + np.sum(gt))
#     un = seg_gt - its
#     dice = its * 2.0 / seg_gt * 100
#     iou = its / un * 100
#
#     return np.around(dice, decimals=2), np.around(iou, decimals=2)
#
#
# def compute(rgb_path, fg_path):
#     img = cv.imread(prefix + rgb_path)
#     m, n, l = img.shape
#     lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
#     fs, tfb = a_b_tfb(lab)
#     X = tfb.reshape((-1, 1))
#     kmeans = KMeans(n_clusters=2, init=np.array([[9.87324814], [27.68430863]]))
#     #     kmeans = KMeans(n_clusters=2, random_state=0)
#     y_hat = kmeans.fit_predict(X)
#     #     print(f'kmeans.cluster_centers_={kmeans.cluster_centers_}')
#     img = y_hat.reshape((m, n))
#
#     img = img * 255
#
#     gt = cv.imread(prefix + fg_path, 0)
#     return DIC_IOU(img, gt)
#
#
# prefix = './Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Tray'
# res = []
# for idx in range(16):
#     name = '{:02d}'.format(idx + 1)
#     rgb_path = f'/Ara2012/ara2012_tray{name}_rgb.png'
#     fg_path = f'/Ara2012/ara2012_tray{name}_fg.png'
#     res.append(list(compute(rgb_path, fg_path)))
#
# for idx in range(27):
#     name = '{:02d}'.format(idx + 1)
#     rgb_path = f'/Ara2013-Canon/ara2013_tray{name}_rgb.png'
#     fg_path = f'/Ara2013-Canon/ara2013_tray{name}_fg.png'
#     res.append(list(compute(rgb_path, fg_path)))
#
# res = np.array(res)
# # print(f'Dice, IOU = {np.mean(res, axis=0)}')
