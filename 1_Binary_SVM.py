# Binary SVM
import numpy as np
np.set_printoptions(threshold=np.nan)
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def svm_classifer(train_images, train_labels):
	rbf_svc = SVC(kernel='linear', gamma=0.1, C=1)
	rbf_svc.fit(train_images, train_labels)
	return rbf_svc

def display_categorization_accuracy(test_labels, test_prediction):
	categorization_accuracy = accuracy_score(test_labels, test_prediction)
	return categorization_accuracy

# Read images
with open('0_bag_of_words_representation_of_images.txt', 'r') as file:
	images = np.array(eval(file.readline()))

# Read labels
with open('0_labels.txt', 'r') as file:
	labels = np.array(eval(file.readline()))

# Shuffle images and labels at a time.
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
X_test = images[dividing_point:]
y_train = labels[:dividing_point]
y_test = labels[dividing_point:]

# Train SVM model
svm_model = svm_classifer(X_train, y_train)

# Predict
y_pred = svm_model.predict(X_test)
Accuracy = display_categorization_accuracy(y_test, y_pred)
print Accuracy

# 60.22
