from __future__ import absolute_import, division, print_function, unicode_literals

# Tested on TensorFlow 1.13.1 and 2.0.0-alpha0

bs = 200  # batch size
num_epochs = 100

# for comparisons
use_RFA = True
use_cnn = False

# Number of features = number of categories for supervised learning
num_feat = 10

import numpy as np
import tensorflow as tf
from tensorflow.linalg import transpose, inv, trace
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout

# prepare the data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000,28,28,1)/255.0
x_test = x_test.reshape(10000,28,28,1)/255.0
y_train = tf.one_hot(y_train, 10)
y_test = tf.one_hot(y_test, 10)


# This network produces the features on images (variable X)
# The one-hot encoding of the labels (variable Y) already are the optimal target features
if use_cnn:
	print("\nUsing a convolutional neural net")

	model = Sequential([
		Conv2D(32, kernel_size=(3, 3), strides=(1,1), input_shape=(28, 28, 1), activation='relu'),
		Conv2D(64, kernel_size=(4, 4), strides=(2,2), activation='relu'),
		Conv2D(64, kernel_size=(3, 3), strides=(1,1), activation='relu'),
		Conv2D(128, kernel_size=(4, 4), strides=(2,2), activation='relu'),
		Flatten(),
		Dense(2048, activation='relu'),
	  	Dense(num_feat) 
	])
else:
	print("\nUsing a 3-layer perceptron")

	model = Sequential([
	  Flatten(input_shape=(28, 28, 1)),
	  Dense(1024, activation='relu'),
	  Dense(1024, activation='relu'),
	  Dense(num_feat) 
	])


D = tf.constant(1e-8 * np.identity(num_feat), tf.float32)

def get_batch_size(X):
	return tf.cast(tf.shape(X)[0], tf.float32)

# produces a matrix which maps the output of the neural net to the actual prediction
def RFA_Pred(F, Y):
	n = get_batch_size(F) 
	K = transpose(F)/n @ F
	A = transpose(F)/n @ Y
	return transpose(A) @ inv(K + D)

def RFA_Loss(F, Y):
	n = get_batch_size(F) 
	K = transpose(F)/n @ F
	A = transpose(F)/n @ Y
	L = transpose(Y)/n @ Y
	return num_feat - trace(inv(K + D) @ A @ inv(L + D) @ transpose(A))

# cross-entropy loss for comparison
def CE_Loss(x,y):
    return tf.keras.backend.categorical_crossentropy(x, y, from_logits=True)

if use_RFA:
	print("and the RFA loss function")

	model.compile(optimizer='adam', loss = RFA_Loss)

	for epoch in range(num_epochs):
		model.fit(x_train, y_train, epochs=1, batch_size=bs) 

		# postprocessing needed to obtain the full prediction model
		F = model.predict(x_train, batch_size=bs)
		P = RFA_Pred(F, y_train)

		# predict the labels of the test data
		y_pred = model.predict(x_test, batch_size=bs) @ transpose(P)

		inacc = 1-tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y_pred, y_test)).numpy()
		print("Epoch %d: test errors: %.2f%%" % (epoch, inacc*100))

else:
	print("and the cross-entropy loss function")

	model.compile(optimizer='adam', loss = CE_Loss)

	for epoch in range(num_epochs):
		model.fit(x_train, y_train, epochs=1, batch_size=bs)

		y_pred = model.predict(x_test, batch_size=bs) 

		inacc = 1-tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y_pred, y_test)).numpy()
		print("Epoch %d: test errors: %.2f%%" % (epoch, inacc*100))

