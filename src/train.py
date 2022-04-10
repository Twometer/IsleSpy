from math import sqrt

import cv2
import json
import numpy as np


# Convert regular HSV to OpenCV HSV
def hsv(h, s, v):
    arr = np.asarray([h / 2, s / 100 * 255, v / 100 * 255]).astype('uint8')
    return arr


def color_correct(img):
    mean_color = cv2.mean(img)
    expected_mean_color = (128, 122, 121)
    multiplier = (
        expected_mean_color[0] / mean_color[0],
        expected_mean_color[1] / mean_color[1],
        expected_mean_color[2] / mean_color[2],
    )
    return (img * multiplier).astype('uint8')


islands = json.load(open("../islands/index.json"))
island_contours = {}

# image = cv2.imread('../images/train/d63ac231f44ab5c5.jpg')  # Reference
# image = cv2.imread('../images/train/2b6bd63b9d574e0d.jpg') # Yellow test
# image = cv2.imread('../images/train/1ebc7a01208733f9.jpg') # Dark test
# image = cv2.imread('../images/train/280e6e8902fae4dd.jpg')  # Green test

island_colors = [
    hsv(64, 9, 70),  # Sand
    hsv(222, 11, 71),  # Sand
    hsv(81, 69, 56),  # Grass
    hsv(148, 65, 15),  # Tree
    hsv(252, 4, 49),  # Rock
]


def img_range(img, center, radius):
    return cv2.inRange(img, center - radius, center + radius)


def normalize_contour(contour, shape):
    points = []
    for inpoints in contour:
        for point in inpoints:
            x = point[0] / shape[0] * 1000
            y = point[1] / shape[1] * 1000
            points.append((round(x), round(y)))
    return points


for i in islands:
    image = cv2.imread(f'../images/train/{i}.jpg')

    image = cv2.bilateralFilter(image, 15, 50, 50)
    image = color_correct(image)

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    seaMask1 = cv2.inRange(image_hsv, hsv(120, 0, 30), hsv(265, 40, 100))
    seaMask2 = cv2.inRange(image_hsv, hsv(12, 0, 50), hsv(320, 15, 100))
    landMask = cv2.bitwise_not(cv2.bitwise_or(seaMask1, seaMask2
                                              ))

    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(4, 4))
    landMask = cv2.morphologyEx(landMask, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, hierarchy = cv2.findContours(image=landMask, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
    for idx in range(0, len(contours)):
        area = cv2.contourArea(contours[idx])
        if sqrt(area) > min(seaMask1.shape[0] / 10, 20):
            cv2.drawContours(image=image, contours=contours, contourIdx=idx, color=(0, 255, 0), thickness=1)
            island_contours[i] = normalize_contour(contours[idx], seaMask1.shape)

json.dump(island_contours, open("../islands/contours.json", "w"))
