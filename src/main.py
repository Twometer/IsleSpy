import json
import cv2
import numpy as np

islands = json.load(open("../islands/index.json"))
contours = json.load(open("../islands/contours.json"))

for contour in contours:
    image = np.zeros((1000, 1000, 1), np.uint8)
    for pt in contours[contour]:
        cv2.circle(image, pt, 1, 255)
    cv2.imshow(contour, image)
    cv2.waitKey()
    cv2.destroyWindow(contour)



print(islands)
