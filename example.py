
bs = 200  # batch size
num_epochs = 20

use_cnn = True

# Number of features = number of categories (for supervised learning)
num_feat = 10

import numpy as np
import tensorflow as tf
from tensorflow.linalg import transpose, inv, trace
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, concatenate, Lambda

# prepare the data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000,28,28,1)/255.0
x_test = x_test.reshape(10000,28,28,1)/255.0
def one_hot(data, nclass): return np.eye(nclass, dtype=np.float32)[data]
y_train = one_hot(y_train, 10)
y_test = one_hot(y_test, 10)

# We need one network producing features for the first data type 
# (the images in this example)
in1 = Input(shape=(28,28,1))
if use_cnn:
    print("\nUsing a convolutional neural net")
    x = Conv2D(32, kernel_size=(3, 3), strides=(1,1), activation='relu')(in1)
    x = Conv2D(64, kernel_size=(4, 4), strides=(2,2), activation='relu')(x)
    x = Conv2D(64, kernel_size=(3, 3), strides=(1,1), activation='relu')(x)
    x = Conv2D(128, kernel_size=(4, 4), strides=(2,2), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)
else:
    print("\nUsing a 3-layer perceptron")
    x = Flatten()(in1)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
out1 = Dense(num_feat)(x)

# and another network producing features for the second data type
# In this example these are the labels, but the one-hot encoding already 
# represents the optimal features, so we just pass it through
in2 = Input(shape=(10,))
out2 = Lambda(lambda x: x)(in2)

# We will use each models separately for predictions...
feat1 = Model(inputs=in1, outputs=out1)
feat2 = Model(inputs=in2, outputs=out2)

# ...but together for training as they have a joint loss function
model = Model(inputs=[in1, in2], outputs=concatenate([out1, out2], axis=1))


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

# produces a matrix inferring X from Y
def inferX(ker, F, X):
    n = get_batch_size(F)
    K, L, A = ker
    return transpose(X)/n @ F @ inv(K + D) @ A @ inv(L + D)

def RFA_Loss(dummy, features):
    F, G = tf.split(features, 2, axis=1)
    return num_feat - relevance(cov(F, G)) 


model.compile(optimizer='adam', loss=RFA_Loss)

# keras really wants us to have a target, although we don't.
dummy = np.zeros(60000)

for epoch in range(1,num_epochs):
    model.fit([x_train, y_train], dummy, epochs=1, batch_size=bs, shuffle=True, verbose=2) 

    print("testing...", end='\r')

    sess = tf.Session()  # because we're not using tf 2.0

    # compute the features on the training images
    F = feat1.predict(x_train, batch_size=bs, verbose=0)
    G = feat2.predict(y_train, batch_size=bs, verbose=0)

    # produce a matrix mapping the output of the model to a prediction 
    # (average over the posterior)
    P = inferY(cov(F, G), G, y_train)

    # label predictions on test data
    tF = feat1.predict(x_test, batch_size=bs, verbose=0)
    y_pred = sess.run(tF @ transpose(P))

    # compute the test loss for good measure
    tG = feat2.predict(y_test, batch_size=bs, verbose=0)
    test_loss = sess.run(num_feat - relevance(cov(tF, tG)))

    sess.close()

    inacc = np.mean(np.argmax(y_pred, axis=-1) != np.argmax(y_test, axis=-1))
    print("Epoch %d: test loss = %.3f  test errors = %.2f%%" % (epoch, test_loss, inacc*100))
    
print("Done.")