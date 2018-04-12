# LMS
import numpy as np

with open('0_bag_of_words_representation_of_images.txt', 'r') as file:
	images = np.array(eval(file.readline()))

# Read labels
with open('0_labels.txt', 'r') as file:
	labels = np.array(eval(file.readline()))

# Shuffle images ad labels at a time.
indices = np.arange(images.shape[0])
np.random.shuffle(indices)
images = images[indices]
labels = labels[indices]

# Normalize images
images = images*1.0
maximum = np.amax(images,1)
maximum = np.reshape(maximum, (maximum.shape[0], 1))
images = images/maximum

# Divide into train and test sets
dividing_point = int(images.shape[0]*0.8)
print dividing_point
X_train = images[:dividing_point]
y_train = labels[:dividing_point]
X_test = images[dividing_point:]
y_test = labels[dividing_point:]

mu = 0.001
w = np.zeros((1,16))

errors = []
#for i in range(dividing_point):
for i in range(dividing_point):
	e_i = y_train[i] - np.matmul(w,np.reshape(X_train[i], (X_train[i].shape[0],1)))
	e_i = np.reshape(e_i, ())
	print e_i
	errors.append(e_i)
	print w
	print X_train[i]
	w = w + mu * int(e_i) * np.reshape(X_train[i], (1, X_train[i].shape[0]))
	
print w.shape
print X_test.shape
y_pred = np.matmul(X_test, np.transpose(w))
print y_pred.shape
print type(y_pred)

'''
function [yd, mse] = lmsFun(u, d, M, N)  
    mu = 0.001;
    w = zeros(1,M);
    for i = 1:N
        e(i) = d(i) - w * u(:,i);
        w = w + mu * e(i) * (u(:,i)');
        mse(i) = abs(e(i))^2;
    end
    yd=(w * u);
    
end
'''

