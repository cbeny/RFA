from __future__ import absolute_import, division, print_function, unicode_literals

# Install TensorFlow
import os
os.system('pip3 install -q tensorflow==2.0.0-alpha0')

import numpy as np
import tensorflow as tf
import tensorflow.linalg as la

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = tf.one_hot(y_train, 10), tf.one_hot(y_test, 10)


print("Using an unregularized 2-hidden layer perceptron")

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(1024, activation='relu'),
  tf.keras.layers.Dense(1024, activation='relu'),
  tf.keras.layers.Dense(10) 
])


def CE_Loss(x,y):
    return tf.keras.backend.categorical_crossentropy(x, y, from_logits=True)

D = tf.constant(1e-8 * np.identity(10), tf.float32)

def cov(F, G):
	n = tf.cast(tf.shape(G)[0], tf.float32)  # batch size
	K = la.matmul(la.transpose(F)/n, F)
	L = la.matmul(la.transpose(G)/n, G)
	A = la.matmul(la.transpose(F)/n, G)
	return K, L, A

def relevance(K, L, A):
	return la.trace(la.inv(K + D) @ A @ la.inv(L + D) @ la.transpose(A))

def inferY(K, L, A, G, Y):
	n = tf.cast(tf.shape(G)[0], tf.float32)
	return la.transpose(Y)/n @ G @ la.inv(L + D) @ la.transpose(A) @ la.inv(K + D)

def RFA_Loss(F, G):
	K, L, A = cov(F, G)
	return 10 - relevance(K, L, A)


use_RFA = True

if use_RFA:
	print("and the RFA loss function")

	model.compile(optimizer='adam', loss = RFA_Loss)

	for epoch in range(10):
		model.fit(x_train, y_train, epochs=1, batch_size=200)

		F = model.predict(x_train, batch_size=200)
		G = y_train
		K, L, A = cov(F, G)
		I = inferY(K, L, A, G, y_train)

		y_pred = model.predict(x_test, batch_size=200) @ la.transpose(I)

		acc = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y_pred, y_test)).numpy()
		print("test accuracy = %.2f%%" % (acc*100))

else:
	print("and the crossentropy loss function")

	model.compile(optimizer='adam', loss = CE_Loss)

	for epoch in range(10):
		model.fit(x_train, y_train, epochs=1, batch_size=200)
		y_pred = model.predict(x_test, batch_size=200) 

		acc = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y_pred, y_test)).numpy()
		print("test accuracy = %.2f%%" % (acc*100))

