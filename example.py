from __future__ import absolute_import, division, print_function, unicode_literals

# Tested on TensorFlow version 2
#import os
#os.system('pip3 install -q tensorflow==2.0.0-alpha0')

# batch size
bs = 200

# for comparisons
use_RFA = True
use_cnn = True

# Number of features = number of categories for supervised learning
num_feat = 10  


import numpy as np
import tensorflow as tf
from tensorflow.linalg import transpose, inv, trace
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

# prepare the data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000,28,28,1)/255.0
x_test = x_test.reshape(10000,28,28,1)/255.0
y_train = tf.one_hot(y_train, 10)
y_test = tf.one_hot(y_test, 10)


# This network produces the features on images (variable X)
# The One-hot encoding of the labels (variable Y) already are the optimal target features
if use_cnn:
	print("Using an unregularized CNN")

	model = Sequential([
		Conv2D(32, kernel_size=(3, 3), strides=(1,1), input_shape=(28, 28, 1), activation='relu'),
		Conv2D(64, kernel_size=(4, 4), strides=(2,2), activation='relu'),
		Conv2D(64, kernel_size=(3, 3), strides=(1,1), activation='relu'),
		Conv2D(128, kernel_size=(4, 4), strides=(2,2), activation='relu'),
		Flatten(),
		Dense(1024, activation='relu'),
	  	Dense(num_feat) 
	])
else:
	print("Using an unregularized 2-hidden layer perceptron")

	model = Sequential([
	  Flatten(input_shape=(28, 28, 1)),
	  Dense(1024, activation='relu'),
	  Dense(1024, activation='relu'),
	  Dense(num_feat) 
	])


D = tf.constant(1e-8 * np.identity(num_feat), tf.float32)

def getbs(X):
	return tf.cast(tf.shape(X)[0], tf.float32)

# computes the covariances between batches of features F of X and G of Y
def cov(F, G):
	n = getbs(G)  # batch size
	K = transpose(F)/n @ F
	L = transpose(G)/n @ G
	A = transpose(F)/n @ G
	return K, L, A

# relevance of features given their covariances
def relevance(K, L, A):
	return trace(inv(K + D) @ A @ inv(L + D) @ transpose(A))

# produces a matrix which maps a vector of features on X to the inferred (expected) value of Y
def inferY(K, L, A, G, Y):
	n = getbs(G)  # batch size
	return transpose(Y)/n @ G @ inv(L + D) @ transpose(A) @ inv(K + D)

def RFA_Loss(F, G):
	K, L, A = cov(F, G)
	return num_feat - relevance(K, L, A)

# cross-entropy loss for comparison
def CE_Loss(x,y):
    return tf.keras.backend.categorical_crossentropy(x, y, from_logits=True)

if use_RFA:
	print("and the RFA loss function")

	model.compile(optimizer='adam', loss = RFA_Loss)

	for epoch in range(100):
		model.fit(x_train, y_train, epochs=1, batch_size=bs)

		# computes the features on the training images
		F = model.predict(x_train, batch_size=bs)

		# the features for the labels are just given by the one-hot encoding of those labels
		G = y_train   

		# convariances of F ang G
		K, L, A = cov(F, G)

		# inferrence matrix
		I = inferY(K, L, A, G, y_train)

		# actual predictions on test data
		y_pred = model.predict(x_test, batch_size=bs) @ transpose(I)

		acc = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y_pred, y_test)).numpy()
		print("test accuracy = %.2f%%" % (acc*100))

else:
	print("and the cross-entropy loss function")

	model.compile(optimizer='adam', loss = CE_Loss)

	for epoch in range(10):
		model.fit(x_train, y_train, epochs=1, batch_size=bs)
		y_pred = model.predict(x_test, batch_size=bs) 

		acc = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y_pred, y_test)).numpy()
		print("test accuracy = %.2f%%" % (acc*100))

