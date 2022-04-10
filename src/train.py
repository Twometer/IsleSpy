import cv2
import json
import numpy as np


# Convert regular HSV to OpenCV HSV
def hsv(h, s, v):
    arr = np.asarray([h / 2, s / 100 * 255, v / 100 * 255]).astype('uint8')
    print(arr)
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


for i in islands:
    image = cv2.imread(f'../images/train/{i}.jpg')

    image = cv2.bilateralFilter(image, 25, 80, 80)
    image = color_correct(image)

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    seaMask = cv2.inRange(image_hsv, hsv(65, 0, 35), hsv(265, 40, 100))
    landMask = cv2.bitwise_not(seaMask)

    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(4, 4))
    landMask = cv2.morphologyEx(landMask, cv2.MORPH_CLOSE, kernel, iterations=4)

    contours, hierarchy = cv2.findContours(image=landMask, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image=image, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2)
    cv2.imshow('Island', image)

    cv2.waitKey()
