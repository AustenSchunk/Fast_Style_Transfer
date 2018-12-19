# Fast Style Transfer with Sparse Convolutions
This is a part of the final project for Computer Vision at Georgia Tech. [Link to Group Project Webpage](https://computervisionproject.github.io/CVFinalUpdate/)

#### -- Project Status: Completed, but may add more features in the future

## Project Intro/Objective
The purpose of this project was to decrease model parameters in order to reduce space and runtime complexities when computing style transfer.

### Main Frameworks Used
* Tensorflow
* Numpy

## Project Description
The general idea behind using model compression in this context is to reduce the number of parameters in order to conserve space and reduce inference speed. There are two main approaches to perform this task, which are using a small-dense network or a large-sparse network. Using the results from [2], which say that a large-sparse network with the same number of parameters will produce higher accuracy in classification, we decided to go with the approach of introducing sparsity into the network proposed in [1]. In order to acheieve sparsity, we used tensorflow's built-in pruning library, which uses the idea of threshold pruning presented in [2]. Unfortunately, tensorflow does not include a sparse convolution operator, so we used the library in [3] that references the techniques presented in [4].

## Experiments and Results
Full report on experiments and results can be found here: [Project Report](https://computervisionproject.github.io/CVFinalUpdate/)

## References
* [1] Johnson et al. [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/pdf/1603.08155.pdf)
* [2] Zhu et al. [Exploring the Efficacy of Pruning for Model Compression](https://arxiv.org/pdf/1710.01878.pdf)
* [3] Hanxiang Hao [Sparse Convolution Op](https://github.com/Connor323/Convolution-with-sparse-kernel-in-TF)
* [4] Han et al. [Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/pdf/1506.02626.pdf)

