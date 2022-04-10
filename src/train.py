import cv2
import json
islands = json.load(open("../islands/index.json"))

image = cv2.imread('../images/train/a010f7cb67b86c07.jpg')
cv2.imshow('test', image)
cv2.waitKey()
