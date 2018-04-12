import os
import cv2
import numpy as np
from scipy.spatial import distance
from sklearn.cluster import KMeans

def find_sift_features(image_path):
	image = cv2.imread(image_path)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	sift = cv2.xfeatures2d.SIFT_create()
	kp, descriptors = sift.detectAndCompute(gray,None)
#	image = cv2.drawKeypoints(gray, kp, None)
#	cv2.imwrite('x.jpg', image)
	return descriptors

def compute_visual_words_k_means(sift_features, k_clusters):
	print np.array(sift_features).shape
	k_means = KMeans(n_clusters = k_clusters)
	k_means.fit(sift_features)
	centroids = k_means.cluster_centers_
	labels = k_means.labels_
	return centroids

def find_similarity(image_sift_feature, visual_words_centroids):
	index = 0
	distances = []
	for feature in visual_words_centroids:
		d = distance.euclidean(feature, image_sift_feature)
		distances.append(d)
	index = distances.index(min(distances))
	return index

def visual_words_representation_of_images(All_sift_features, number_of_features_in_each_image, visual_words_centroids, k_clusters):
	images = []
	count = 0
	i = 0
	image_feature = [0]*k_clusters
	for image_sift_feature in All_sift_features:
		index = find_similarity(image_sift_feature, visual_words_centroids)
		image_feature[index] += 1
		count+=1
		if count == number_of_features_in_each_image[i]:
			images.append(image_feature)
			image_feature = [0]*k_clusters
			count = 0
			i+=1
	return images

# Directories
directories = ['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding']

# All image names and labels
image_names = []
labels = []
sift_features = []
number_of_features_in_each_image = []
main_path = '../image_data/'
label = 0
for dir in directories:
	dirpath = main_path + dir
	filepaths = os.listdir(dirpath)
	for path in filepaths:
		filepath = dirpath + '/' + path
		print filepath
		image_names.append(filepath)
		labels.append(label)
		features = find_sift_features(filepath)
		number_of_features_in_each_image.append(features.shape[0])
		for feature in features:
			sift_features.append(list(feature))
	label+=1
print len(sift_features)
print number_of_features_in_each_image

# Use K-means to compute visual words # Cluster descriptors
k_clusters = 16
visual_words_centroids = compute_visual_words_k_means(sift_features, k_clusters)
print visual_words_centroids.shape

# Representation of images in terms of visual words
images = visual_words_representation_of_images(sift_features, number_of_features_in_each_image, visual_words_centroids, k_clusters)
print images
print len(images)
print np.array(images).shape

# Save
with open('image_names.txt', 'w') as file:
	file.write(str(image_names))

with open('bag_of_words_representation_of_images.txt', 'w') as file:
	file.write(str(images))
