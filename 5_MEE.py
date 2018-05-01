# MEE

import numpy as np
np.set_printoptions(threshold=np.nan)
from sklearn.metrics import accuracy_score

def display_categorization_accuracy(test_labels, test_prediction):
	categorization_accuracy = accuracy_score(test_labels, test_prediction)
	return categorization_accuracy

with open('0_bag_of_words_representation_of_images.txt', 'r') as file:
	images = np.array(eval(file.readline()))

# Read labels
with open('0_labels.txt', 'r') as file:
	labels = np.array(eval(file.readline()))

# Shuffle images ad labels at a time.
indices = np.arange(images.shape[0])
np.random.shuffle(indices)
images = images[indices]
labels = labels[indices] + 1

# Normalize images
images = images*1.0
maximum = np.amax(images,1)
maximum = np.reshape(maximum, (maximum.shape[0], 1))
images = images/maximum

# Divide into train and test sets
dividing_point = int(images.shape[0]*0.8)
X_train = images[:dividing_point]
y_train = labels[:dividing_point]
X_test = images[dividing_point:]
y_test = labels[dividing_point:]

mu = 0.01
sigma = 20
w = np.zeros((1,len(X_train[0])))
k = 500

for i in range(dividing_point):
	num = 0
	den = 0
	e_i = y_train[i] - np.matmul(w,np.reshape(X_train[i], (X_train[i].shape[0],1)))
	e_i = np.reshape(e_i, ())
	for j in range(max(0, i-k), i):
		e_j = y_train[j] - np.matmul(w,np.reshape(X_train[j], (X_train[j].shape[0],1)))
		e_j = np.reshape(e_j, ())
		k_n = np.exp((-1.0)*((e_i-e_j)**2)*(1.0/sigma))
		den += den + k_n
		num += num + k_n * (-2.0) * (1.0/sigma) * (e_i-e_j) * (X_train[i] - X_train[j])
	#print i, num, den
	if den!=0:
		w = w + mu * (num*1.0/den*1.0)
print w

y_pred = np.matmul(X_test, np.transpose(w))
y_pred = np.round(y_pred) * -1
print y_pred
Accuracy = display_categorization_accuracy(y_test, y_pred)
print Accuracy
