# Relevant Feature Analysis

This is an example of implementation of the approach presented in https://arxiv.org/abs/1904.10387

example.jl is a minimal working example, which can be used to reproduce Fig. 1 (top row). It trains and tests a neural network for the classification of MNIST digits using a novel loss function, coupled with an extra post-processing operation. This code can be straighforwardly extended to learn to do inference between two correlated datasets. 

cifar.jl can be used to reproduce the bottow row of Fig. 1.
