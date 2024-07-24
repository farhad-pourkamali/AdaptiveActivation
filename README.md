### Overview

In the realm of neural network architecture, the choice of activation functions plays a critical role in enabling the network to model complex non-linear relationships inherent in the data. A promising area of research within this domain is the exploration of **adaptive or trainable activation functions**. These dynamic functions stand in contrast to conventional static activation functions like ReLU and Sigmoid, which do not change their behavior during training. Adaptive activation functions have the unique capability to adjust their parameters in response to the evolving data distribution and the learning trajectory of the model. This adaptability allows for a more flexible and potentially more effective learning process, as the activation function can optimize its behavior to better suit the specific problem at hand.

### Repository Contents
This repository offers a TensorFlow/Keras-based implementation of three well-regarded activation functions: the Exponential Linear Unit (ELU), Softplus, and Swish. Our focus extends to two distinct approaches in deploying adaptive activation functions within a neural network:

+ Shared Parameters Approach: This configuration employs a common set of parameters for the adaptive activation functions across all neurons in a particular hidden layer, promoting parameter efficiency and reducing the risk of overfitting.

+ Individual Parameters Approach: In contrast, this setup allows each neuron to adapt its activation function independently by learning its own set of parameters, providing the potential for a highly tailored and nuanced modeling of complex patterns.

Our implementations aim to provide a practical framework for experimenting with and integrating adaptive activation functions into neural network models, paving the way for enhanced performance and deeper insights into the learning dynamics of neural networks.

If you use our implementation of adaptive activation functions, please cite the following work: 

+ Pourkamali-Anaraki, F., Nasrin, T., Jensen, R. E., Peterson, A. M., & Hansen, C. J. Adaptive Activation Functions for Predictive Modeling with Sparse Experimental Data. arXiv preprint. 

<img src="https://raw.githubusercontent.com/farhad-pourkamali/AdaptiveActivation/main/adaptive.png" alt="adaptive" width=1000 align=center />

If you use this implementation, please cite the following paper: 
Pourkamali-Anaraki, F., Nasrin, T., Jensen, R.E. et al. Adaptive activation functions for predictive modeling with sparse experimental data. Neural Comput & Applic (2024). https://doi.org/10.1007/s00521-024-10156-8


