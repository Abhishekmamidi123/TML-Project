# LMS
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

mu = 0.1
w = np.zeros((1,16))

errors = []
#for i in range(dividing_point):
for i in range(dividing_point):
	e_i = y_train[i] - np.matmul(w,np.reshape(X_train[i], (X_train[i].shape[0],1)))
	e_i = np.reshape(e_i, ())
	errors.append(e_i)
	w = w + mu * int(e_i) * np.reshape(X_train[i], (1, X_train[i].shape[0]))

y_pred = np.matmul(X_test, np.transpose(w))
y_pred = np.round(y_pred)

Accuracy = display_categorization_accuracy(y_test, y_pred)
print Accuracy
