# Relevant Feature Analysis

This is an example of implementation of the approach presented in https://arxiv.org/abs/1904.10387v3

`example.jl` is a minimal working example of RFA, which learns the correlations between two datasets. In this example it is applied to supervised classification on MNIST, but the code is designed so that it can be straighforwardly extended to the general unsupervised setting. This can be used to reproduce the top row of Fig. 1.

`example.py` is a tensorflow implementation of the above.

`mnist.py` does the same, but is slightly streamlined for supervised classification. In this setting, RFA boils down to a new loss function, coupled with an extra post-processing operation.

`cifar.jl` can be used to reproduce the bottow row of Fig. 1.
