# Feature extraction of images
import cv2
import numpy as np

#directories = ['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding']

#for dir in directories:
#	filepath = 'image_data'
	
path = 'art294.jpg'
image = cv2.imread(path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(gray,None)

image = cv2.drawKeypoints(gray, kp, None)
cv2.imwrite('x.jpg', image)
print des
print des.shape
