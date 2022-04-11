import json
import cv2
import numpy as np

islands = json.load(open("../islands/index.json"))
contours = json.load(open("../islands/contours.json"))


# for contour in contours:
#     image = np.zeros((1000, 1000, 1), np.uint8)
#     for pt in contours[contour]:
#         cv2.circle(image, pt, 1, 255)
#     cv2.imshow(contour, image)
#     cv2.waitKey()
#     cv2.destroyWindow(contour)

# offset_x = int(image.shape[0] * 0.25)
# offset_y = int(image.shape[1] * 0.05)
# cropped = image[offset_y:-offset_y, offset_x:-offset_x]

# Convert regular HSV to OpenCV HSV
def hsv(h, s, v):
    arr = np.asarray([h / 2, s / 100 * 255, v / 100 * 255]).astype('uint8')
    return arr


image = cv2.imread('../images/validate/f.jpg')
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# beachMask = cv2.inRange(image_hsv, hsv(12, 0, 50), hsv(320, 15, 100))
seaMask = cv2.inRange(image_hsv, hsv(190, 25, 10), hsv(220, 65, 60))
paperMask = cv2.inRange(image_hsv, hsv(40, 25, 10), hsv(60, 80, 35))

mask = cv2.bitwise_or(seaMask, paperMask)

cv2.imshow("Test", image)
cv2.imshow("Mask", mask)
cv2.waitKey()

print(islands)
