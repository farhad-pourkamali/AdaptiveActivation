import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense
from sklearn.metrics import accuracy_score
from scipy.io import savemat, loadmat

# Create a custom ELU activation function with a learnable parameter 
class CustomActivationELU(tf.keras.layers.Layer):
    def __init__(self, units=1, **kwargs):
        super(CustomActivationELU, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.alpha = self.add_weight(
            shape=(self.units,),
            initializer='ones',
            trainable=True,
            name='alpha',
        )
        super(CustomActivationELU, self).build(input_shape)

    def call(self, x):
        return tf.where(x > 0, x, self.alpha[0] * (tf.math.exp(x) - 1)) 
    
# Create a custom Softplus activation function with a learnable parameter 
class CustomActivationSoftPlus(tf.keras.layers.Layer):
    def __init__(self, units=1, **kwargs):
        super(CustomActivationSoftPlus, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.beta = self.add_weight(
            shape=(self.units,),
            initializer='ones',
            trainable=True,
            name='beta',
        )
        super(CustomActivationSoftPlus, self).build(input_shape)

    def call(self, x):
        return  tf.math.log(tf.math.pow(self.beta[0], 2) + tf.math.exp(x))  
    
# Create a custom Swish activation function with a learnable parameter 
class CustomActivationSwish(tf.keras.layers.Layer):
    def __init__(self, units=1, **kwargs):
        super(CustomActivationSwish, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(self.units,),
            initializer='ones',
            trainable=True,
            name='gamma',
        )
        super(CustomActivationSwish, self).build(input_shape)

    def call(self, x):
        return x * tf.math.sigmoid(self.gamma[0] * x) 
    
# create a model with "identical" activation functions
def build_model_identical(num_input, num_hidden, num_output, activation, seed):
    """
    num_input: number of input features 
    num_hidden: number of units in the hidden layer 
    num_output: number of classes 
    """
    model = keras.Sequential([
    Input(shape=(num_input,)),
    Dense(num_hidden, use_bias=True, activation=activation,
                      kernel_initializer = tf.keras.initializers.HeUniform(seed)),
    Dense(num_output, use_bias=True, activation='softmax',
                      kernel_initializer = tf.keras.initializers.HeUniform(seed))
                      ])
    
    return model


# create a model with "different" activation functions 
def build_model_different(num_input, num_hidden, num_output, bias, activation, seed):
    """
    num_input: number of input features 
    num_hidden: number of units in the hidden layer 
    num_output: number of classes 
    """
    inputs_ = Input(shape=(num_input,), name='input')
    hidden_list = []
    if activation == 'ELU':
        for num in range(num_hidden):
            hidden_list.append(Dense(1, activation=CustomActivationELU(units=1), 
                            use_bias=bias, name='unit' + str(num), 
                            kernel_initializer = tf.keras.initializers.HeUniform(seed))(inputs_))
    
        concat_layer = tf.keras.layers.Concatenate()
        concat = concat_layer(hidden_list)

    elif activation == 'Softplus':
        for num in range(num_hidden):
            hidden_list.append(Dense(1, activation=CustomActivationSoftPlus(units=1), 
                            use_bias=bias, name='unit' + str(num), 
                            kernel_initializer = tf.keras.initializers.HeUniform(seed))(inputs_))
    
        concat_layer = tf.keras.layers.Concatenate()
        concat = concat_layer(hidden_list)

    elif activation == 'Swish':
        for num in range(num_hidden):
            hidden_list.append(Dense(1, activation=CustomActivationSwish(units=1), 
                            use_bias=bias, name='unit' + str(num), 
                            kernel_initializer = tf.keras.initializers.HeUniform(seed))(inputs_))
    
        concat_layer = tf.keras.layers.Concatenate()
        concat = concat_layer(hidden_list)


    outputs_ = Dense(num_output, activation="softmax", use_bias=True, name='output',
                 kernel_initializer = tf.keras.initializers.HeUniform(seed))(concat)
    
    model = tf.keras.Model(inputs=inputs_, outputs=outputs_)

    return model


def find_cov_unc(X_train, y_train, X_test, y_test, model, alpha):
    """
    Return empirical coverage and uncertainty score using conformal prediction
    with the target coverage level 1-alpha 
    """

    # Get calibration scores 
    y_prob = model.predict(X_train)

    n = X_train.shape[0]

    cal_scores = 1 - y_prob[np.arange(n), y_train]

    q_level = np.ceil((n+1)*(1-alpha))/n

    qhat = np.quantile(cal_scores, q_level, method='higher')

    prediction_sets = model.predict(X_test) >= (1-qhat) 

    # coverage
    cov = prediction_sets[np.arange(X_test.shape[0]), y_test].mean()

    # width 
    array = prediction_sets.sum(axis=1) 
    unc = array[array != 0].mean() 

    return cov, unc 