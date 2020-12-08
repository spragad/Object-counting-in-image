"""
@author: Pragadeesh
"""
import numpy as np
import cv2
import time

print("Reading the image....")
time.sleep(1)
img=cv2.imread('C:/Users/Pragadeesh/Desktop/Resolute Ai/Task_3.jpeg')

#Convert to grey scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Increasing gamma in grey scale
gray_gamma = np.array(255 * (gray / 255) ** 1.2 , dtype='uint8')

#Equalized Grey scale
gray_equ = cv2.equalizeHist(gray_gamma)

#Thresholding to highlight pipe edges
hil = cv2.adaptiveThreshold(gray_equ, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255, 19)
hil = cv2.bitwise_not(hil)
print("Processing the image....")
time.sleep(1)
#clean all noise using dilatation and erosion
kernel = np.ones((10,10), np.uint8)
img_dil = cv2.dilate(hil, kernel, iterations=1)
img_er = cv2.erode(img_dil,kernel, iterations=1)
img_pro = cv2.medianBlur(img_er, 7)
inv = cv2.bitwise_not(img_pro)

print("Counting the number of pipes....")
time.sleep(1)
#Labeling and counting
ret, labels = cv2.connectedComponents(inv)
label_hue = np.uint8(179 * labels / np.max(labels))
blank_ch = 255 * np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
labeled_img[label_hue == 0] = 0

print('Number Of Pipes =', ret-1)

cv2.imshow('result',labeled_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

