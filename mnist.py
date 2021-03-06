# tested on tensorflow 2.2

bs = 200  # batch size
num_epochs = 10

# for comparisons
use_RFA = True
use_cnn = True

# Number of features = number of categories for supervised learning
num_feat = 10

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout

# prepare the data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000,28,28,1)/255.0
x_test = x_test.reshape(10000,28,28,1)/255.0
def one_hot(data, nclass): return np.eye(nclass, dtype=np.float32)[data]
y_train = one_hot(y_train, 10)
y_test = one_hot(y_test, 10)

# We need a neural net as usual, with 1 linear output neuron per class
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

def RFA_Pred(F, Y):
	n = get_batch_size(F) 
	K = tf.matmul(F/n, F, transpose_a=True)
	A = tf.matmul(F/n, Y, transpose_a=True)
	return tf.matmul(A, tf.linalg.inv(K + D), transpose_a=True)

def RFA_Loss(F, Y):
	n = get_batch_size(F) 
	K = tf.matmul(F/n, F, transpose_a=True)
	A = tf.matmul(F/n, Y, transpose_a=True)
	L = tf.matmul(Y/n, Y, transpose_a=True)
	return num_feat - tf.linalg.trace(A @ tf.linalg.inv(L) @ tf.linalg.matrix_transpose(A) @ tf.linalg.inv(K + D))

# cross-entropy loss for comparison
def CE_Loss(x,y):
    return tf.keras.backend.categorical_crossentropy(x, y, from_logits=True)

if use_RFA:
	print("and the RFA loss function")

	model.compile(optimizer='adam', loss = RFA_Loss)

	for epoch in range(num_epochs):
		model.fit(x_train, y_train, epochs=1, batch_size=bs, shuffle=True) 

		# postprocessing needed to obtain the full prediction model
		features = model.predict(x_train, batch_size=bs)
		P = RFA_Pred(features, y_train)

		# predict the labels of the test data
		y_pred = model.predict(x_test, batch_size=bs) @ np.transpose(P)

		inacc = 1-tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y_pred, y_test)).numpy()
		print("Epoch %d: test errors: %.2f%%" % (epoch, inacc*100))

else:
	print("and the cross-entropy loss function")

	model.compile(optimizer='adam', loss = CE_Loss)

	for epoch in range(num_epochs):
		model.fit(x_train, y_train, epochs=1, batch_size=bs)

		y_pred = model.predict(x_test, batch_size=bs) 
		inacc = np.mean(np.argmax(y_pred, axis=-1) != np.argmax(y_test, axis=-1))
		print("Epoch %d: test errors: %.2f%%" % (epoch, inacc*100))

