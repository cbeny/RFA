from __future__ import absolute_import, division, print_function, unicode_literals

# Tested on TensorFlow 1.13.1 and 2.0.0-alpha0

bs = 200  # batch size
num_epochs = 100

use_cnn = True

# Number of features = number of categories (for supervised learning)
num_feat = 10

import numpy as np
import tensorflow as tf
from tensorflow.linalg import transpose, inv, trace
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, concatenate

# prepare the data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000,28,28,1)/255.0
x_test = x_test.reshape(10000,28,28,1)/255.0
y_train = tf.one_hot(y_train, 10)
y_test = tf.one_hot(y_test, 10)

# This network produces the features on images
in1 = Input(shape=(28,28,1))
if use_cnn:
	print("\nUsing a convolutional neural net")
	aux = Conv2D(32, kernel_size=(3, 3), strides=(1,1), activation='relu')(in1)
	aux = Conv2D(64, kernel_size=(4, 4), strides=(2,2), activation='relu')(aux)
	aux = Conv2D(64, kernel_size=(3, 3), strides=(1,1), activation='relu')(aux)
	aux = Conv2D(128, kernel_size=(4, 4), strides=(2,2), activation='relu')(aux)
	aux = Flatten()(aux)
	aux = Dense(2048, activation='relu')(aux)
else:
	print("\nUsing a 3-layer perceptron")
	aux = Flatten()(in1)
	aux = Dense(1024, activation='relu')(aux)
	aux = Dense(1024, activation='relu')(aux)
out1 = Dense(num_feat)(aux)

# out2 should be given by another trainable neural net acting on in2
# but in this example, one-hot encoding already represents the optimal features
in2 = Input(shape=(10,))
out2 = in2

# We will use each models separately for predictions...
feat1 = Model(inputs=in1, outputs=out1)
feat2 = Model(inputs=in2, outputs=out2)

# ...but together for training as they have a joint loss function
model = Model(inputs=[in1, in2], outputs=concatenate([out1, out2]))


D = tf.constant(1e-8 * np.identity(num_feat), tf.float32)

def get_batch_size(X):
	return tf.cast(tf.shape(X)[0], tf.float32)

# computes the covariances between batches of features F of X and G of Y
def cov(F, G):
	n = get_batch_size(G) 
	K = transpose(F)/n @ F
	L = transpose(G)/n @ G
	A = transpose(F)/n @ G
	return K, L, A

# relevance of features given their covariances
def relevance(ker):
	K, L, A = ker
	return trace(inv(K + D) @ A @ inv(L + D) @ transpose(A))

# produces a matrix which maps a vector of features on X to the inferred (expected) value of Y
def inferY(ker, G, Y):
	n = get_batch_size(G)  
	K, L, A = ker
	return transpose(Y)/n @ G @ inv(L + D) @ transpose(A) @ inv(K + D)

def RFA_Loss(dummy, features):
	F, G = tf.split(features, 2, axis=1)
	return num_feat - relevance(cov(F, G)) 

	print("and the RFA loss function")

model.compile(optimizer='adam', loss=RFA_Loss)

# keras really wants us to have a target, but we don't.....
dummy = [0.0 for i in range(60000)] 

for epoch in range(num_epochs):
	model.fit([x_train, y_train], dummy, epochs=1, batch_size=bs) 

	# computes the features on the training images
	F = feat1.predict(x_train, batch_size=bs)
	G = feat2.predict(y_train, batch_size=bs)

	# produces a matrix I mapping the output of the model to a prediction 
	# (average over the posterior)
	ker = cov(F, G)
	I = inferY(ker, G, y_train)

	# label predictions on test data
	y_pred  = feat1.predict(x_test, batch_size=bs) @ transpose(I)
	y_true  = feat2.predict(y_test, batch_size=bs)

	inacc = 1-tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y_pred, y_true)).numpy()
	print("Epoch %d: test errors: %.2f%%" % (epoch, inacc*100))
