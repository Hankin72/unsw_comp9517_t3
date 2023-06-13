#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 23:05:46 2020

@author:
by Haojin Guo z5216214
"""

# ......IMPORT .........
import math
import argparse
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import copy
from scipy import ndimage as ndi
from warnings import warn
from collections import Counter
import random


def task1(args, img):
    # get matrix pixel values of row and column
    height, width = img.shape
    # iso-data intensity thresholding, get an arbitrary int t
    t_new = np.random.randint(0, 256)
    # print("the first t", t_new)
    t_old = 0
    t_list = []
    t_list.append(t_new)
    while abs(t_new - t_old) >= 0.005:
        output_img = copy.deepcopy(img)
        # background encoded as white[pixel value: 255], the rice kernels encoded as back[pixel value: 255],
        rice_kernels = (img < t_new)
        background = (img > t_new)
        # u0 mean of intensity values < t_temp-----rice kernels , u1 = mean of intensity values >= t_temp --- background
        u0 = np.mean(img[rice_kernels])
        u1 = np.mean(img[background])
        t_old = t_new
        t_new = round((u0 + u1) / 2)
        t_list.append(t_new)
    # get threshold_value
    t = t_list[-1]
    # print(t_list)
    # print(len(t_list))
    # plot the threshold value t at every iteration on a graph
    x = [i for i in range(len(t_list))]
    y = [j for j in t_list]
    plt.plot(x, y, 's-', color='r', label="t_Value")  # s-:方形
    plt.xlabel("Iteration times of t value")
    plt.ylabel("The value of t")
    plt.show()
    # get binary_image
    for h in range(height):
        for w in range(width):
            if img[h, w] >= t:
                output_img[h, w] = 0
            else:
                output_img[h, w] = 255
    return output_img, t


def task2(args, img, selem=None, out=None, mode='nearest', cval=0.0, ):
    # the process of median blur
    if selem is None:
        selem = ndi.generate_binary_structure(img.ndim, img.ndim)
    median_blur_img = ndi.median_filter(img, footprint=selem, output=out, mode=mode, cval=cval)
    plt.imshow(median_blur_img, 'gray')
    plt.title(f'Filtered binary image'), plt.xticks([]), plt.yticks([])
    plt.show()

    # two_pass pre_processing
    img_borden = cv2.copyMakeBorder(median_blur_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)

    height, width = img_borden.shape

    neighborhoods_1 = [[-1, -1], [-1, 0], [-1, 1],
                       [0, -1]]

    label_img = np.zeros_like(img_borden)

    no_of_label = 1
    label_dict = {}

    # first_pass
    for h in range(1, height - 1):
        for w in range(1, width - 1):

            if img_borden[h, w] != 0:
                continue
                # collect label in a list

            label_list = []
            # check the neighborhoods of pixel value

            for neighbor in neighborhoods_1:
                if label_img[h + neighbor[0], w + neighbor[-1]] != 0:
                    label_list.append(label_img[h + neighbor[0], w + neighbor[-1]])

            if len(label_list) == 0:
                label_img[h, w] = no_of_label
                no_of_label += 1
            else:
                label_img[h, w] = np.min(label_list)
                # combine
                if np.min(label_list) not in label_dict:

                    label_dict[np.min(label_list)] = []
                    for value in label_list:
                        label_dict[np.min(label_list)].append(value)
                else:
                    for value in label_list:
                        label_dict[np.min(label_list)].append(value)

    for key in label_dict.keys():
        label_dict[key] = set(label_dict[key])

    # print(len(label_dict.keys()))
    # print(label_dict)







    # second pass

    neighborhoods_2 = [[-1, -1], [-1, 0], [-1, 1],
                       [0, -1], [0, 0], [0, 1],
                       [1, -1], [1, 0], [1, 1]]

    row, column = label_img.shape
    # print(row, column)

    # np.random.randint(0, 256)

    for r in range(1, row - 1):
        for c in range(1, column - 1):
            if label_img[r, c] != 0:
                # check the neighborhoods of pixel value
                for neighbor in neighborhoods_2:
                    if label_img[r + neighbor[0], c + neighbor[-1]] != 0:
                        label_img[r + neighbor[0], c + neighbor[-1]] = label_img[r, c]

    for i in range(row * column):
        r = np.random.randint(1, row - 1)
        c = np.random.randint(1, column - 1)
        if label_img[r, c] != 0:
            # check the neighborhoods of pixel value
            for neighbor in neighborhoods_2:
                if label_img[r + neighbor[0], c + neighbor[-1]] != 0:
                    label_img[r + neighbor[0], c + neighbor[-1]] = label_img[r, c]

    #
    # kernel = dict(Counter(temp)-Counter(label_dict))
    # print(kernel)

    counts = {}
    for r in range(row):
        for c in range(column):
            if label_img[r, c] == 0:
                continue
            for key in label_dict.keys():
                if label_img[r, c] in label_dict[key]:
                    label_img[r, c] = key
            l = label_img[r, c]
            if l not in counts.keys():
                counts[l] = 1
            else:
                counts[l] += 1

    number_kernels = 0

    return median_blur_img, counts, label_img


def task3(args, image, counts):
    nums = 0
    values = []
    new_counts = {}

    for item in counts.values():
        values.append(item)
    sift = max(values) / 2
    for k, v in counts.items():
        if counts[k] > sift:
            nums += 1
            new_counts[k] = v

    # print(new_counts.keys())
    percentage = nums / len(counts.keys())


    row, column = image.shape
    for r in range(row):
        for c in range(column):
            if image[r, c] == 0:
                continue
            # for key in new_counts:
            for key in new_counts.keys():
                if image[r, c] == key:
                    image[r, c] = 0
                else:
                    image[r, c] = 255

    # plt.imshow(image, 'gray')
    # plt.show()

    visited = np.zeros(image.shape, dtype=bool)
    labels = np.zeros(image.shape, dtype=int)
    count = 1
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if not visited[i, j]:
                if image[i, j] == 0:
                    # bfs
                    queue = []
                    queue.append((i, j))
                    while queue:
                        p = queue.pop(0)
                        if p[0] < 0 or p[0] >= image.shape[0]:
                            continue
                        if p[1] < 0 or p[1] >= image.shape[1]:
                            continue
                        if visited[p[0], p[1]]:
                            continue
                        if image[p[0], p[1]] != 0:
                            continue

                        labels[p[0], p[1]] = count

                        up = (p[0], p[1] - 1)
                        queue.append(up)
                        down = (p[0], p[1] + 1)
                        queue.append(down)

                        left = (p[0] - 1, p[1])
                        queue.append(left)
                        right = (p[0] + 1, p[1])
                        queue.append(right)

                        up_left = (p[0] - 1, p[1] - 1)
                        queue.append(up_left)
                        down_right = (p[0] + 1, p[1] + 1)
                        queue.append(down_right)

                        up_right = (p[0] - 1, p[1] + 1)
                        queue.append(up_right)
                        down_left = (p[0] + 1, p[1] - 1)
                        queue.append(down_left)

                        visited[p[0], p[1]] = True

                    count += 1
                else:
                    visited[i, j] = True
    count -= 1

    return percentage, labels


my_parser = argparse.ArgumentParser()
my_parser.add_argument('-o', '--OP_folder', type=str, help='Output folder name', default='OUTPUT')
# /Users/guohaojin/Downloads/COMP9517/ass01/ass01_output
my_parser.add_argument('-m', '--min_area', type=int, action='store', required=False,
                       help='Minimum pixel area to be occupied, to be considered a whole rice kernel')

my_parser.add_argument('-f', '--input_filename', type=str, action='store', required=False, help='Filename of image ')

# Execute parse_args()
args = my_parser.parse_args()


# img1 = cv2.imread(args.input_filename, 0)

img1 = cv2.imread('rice_img1.png', 0)

plt.imshow(img1, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.show()

binary_img, threshold_value = task1(args, img1)
print(f"The threshold value is {threshold_value}. ")
plt.imshow(binary_img, 'gray'), plt.xticks([]), plt.yticks([])
plt.title(f'Threshold Value ={threshold_value}.')
plt.show()

median_blur_img, counts, label_img = task2(args, binary_img)
print(f"The number of rice kernels in the image are {len(counts.keys())}.")

percentage_damged_rice, labels = task3(args, label_img, counts)
print(f'The percentage of damaged rice kernels are {round(percentage_damged_rice * 100, 2)}%.')

plt.imshow(labels, 'gray'), plt.xticks([]), plt.yticks([])
plt.title("Excluding all damaged kernels---rice")
plt.show()


# plt.imshow(median_blue_img, 'gray')
# plt.title(f'Filtered binary image'), plt.xticks([]), plt.yticks([])
# plt.show()

# if __name__ == "__main__":
