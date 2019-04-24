# Relevant Feature Analysis

This program trains and tests a neural network for the classification of MNIST digits using a new strategy presented in https://arxiv.org/abs/1904.10387

The difference with standard approaches is a new loss function, as well as an extra post-processing linear operation.

It converges very fast, and has a strong regularizating effect.

This code can also be straighforwardly extended to learn to do inference between two correlated datasets.
