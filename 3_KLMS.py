# KLMS

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

# Training - Store errors
mu = 0.001
sigma = 2
errors = []
e_i = y_train[0]
errors.append(e_i)
for i in range(1, dividing_point):
	sum = 0
	for j in range(i):
		num = (np.linalg.norm(X_train[j]-X_train[i])**2)*(-1.0)
		den = sigma * sigma * 1.0
		sum = sum + (errors[j] * np.exp(num/den))
	y_i = mu*sum
	e_i = y_train[i] - y_i
	errors.append(e_i)

y_pred = []
for i in range(len(X_test)):
	sum = 0
	for j in range(len(errors)):
		num = (np.linalg.norm(X_train[j]-X_test[i])**2)*(-1.0)
		den = sigma * sigma * 1.0
		sum = sum + (errors[j] * np.exp(num/den))
	y_i = mu*sum
	y_pred.append(np.round(y_i))

Accuracy = display_categorization_accuracy(y_test, y_pred)
print Accuracy
