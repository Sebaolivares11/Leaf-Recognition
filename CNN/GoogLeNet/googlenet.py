#!/usr/bin/env python
# coding: utf-8

# # Implementing GoogLeNet in Tensorflow.

# In[2]:


# importing pakages
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout,Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import plot_model
import pydot


# ## 1 - The Inception block

# In[3]:


def inception_block(X, filters, stage, block):
    """
    Implementation of the inception block as defined in paper 
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    filters -- python list of integers, defining the number of filters in the CONV layers 
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the inception block
    """
    # Retrieve Filters 
    # F1, F3, F5 stand for number of filter used for repective convolution
    # F13 and F15 are stands for the number of 1*1 filters in the reduction layer used before the 3*3 and 5*5 convolutions
    # FP is stands for the number of 1*1 filters after max pooling
    F1, F13, F3, F15, F5, FP = filters
    
    # defining name basis
    conv_name_base = 'conv' + str(stage) + block + '/'
    p_name_base = 'p' + str(stage) + block + '/'
    
    # 1x1 filter on output of previos layer 
    inception_1x1 = Conv2D(filters=F1, kernel_size=(1,1), padding='same', activation='relu', name=conv_name_base+'1x1')(X)
    
    # 1x1 filter on output of previos layer for reduction of dimension before 3x3 and then 3x3 
    inception_3x3_reduce = Conv2D(filters=F13, kernel_size=(1,1), padding='same', activation='relu', name=conv_name_base+'3x3_reduce')(X)
    inception_3x3 = Conv2D(filters=F3, kernel_size=(3,3), padding='same', activation='relu', name=conv_name_base+'3x3')(inception_3x3_reduce)
    
    # 1x1 filter on output of previos layer for reduction of dimension before 5x5 and then 5x5 
    inception_5x5_reduce = Conv2D(filters=F15, kernel_size=(1,1), padding='same', activation='relu', name=conv_name_base+'5x5_reduce')(X)
    inception_5x5 = Conv2D(filters=F5, kernel_size=(5,5), padding='same', activation='relu', name=conv_name_base+'5x5')(inception_5x5_reduce)
    
    # pooling layer on output of previos layer then on this 1x1 filter for channel reduction.    
    inception_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1), padding='same', name=p_name_base+'pool')(X)
    inception_1x1_afterpool = Conv2D(filters=FP, kernel_size=(1,1), padding='same', activation='relu', name=conv_name_base+'1x1_pool')(inception_pool)

    X = tf.keras.layers.concatenate([inception_1x1, inception_3x3, inception_5x5, inception_1x1_afterpool], axis = 3)
    
    return X


# ## 2 - Building Model(GoogLeNet)

# In[10]:


def GoogLeNet(input_shape = (224, 244, 3), classes=1000, weight_path=None):
    """
    Implementation of the popular GoogLeNet with architecture mention in his original paper:
    CONV2D -> MaxPool -> LocalRespNorm -> CONV2D -> CONV2D -> LocalRespNorm -> MaxPool-> INCEPTION_BLOCK*2
    -> MaxPool -> INCEPTION_BLOCK*5 -> MaxPool-> INCEPTION_BLOCK*2 -> AVERAGEPOOL -> FC -> SOFTMAX

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
    X = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2),padding='same', activation='relu')(X_input)  # output (11211264)
    X = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(X)                                  # output 565664
    X = Conv2D(filters=64, kernel_size=(1,1), strides=(1,1),padding='same', activation='relu')(X)        # output 565664
    X = Conv2D(filters=192, kernel_size=(3,3), strides=(1,1),padding='same', activation='relu')(X)       # output 5656192
    X = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(X)                                  # 2828192
    X = inception_block(X, filters = [64, 96, 128, 16, 32, 32], stage = 3, block = 'a')               # 2828256  
    X = inception_block(X, filters = [128, 128, 192, 32, 96, 64], stage = 3, block = 'b')             # 2828480
    X = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(X)                                  # 1414480
    a4 = inception_block(X, filters = [192, 96, 208, 16, 48, 64], stage = 4, block = 'a')              # 1414512
    X = inception_block(a4, filters = [160, 112, 224, 24, 64, 64], stage = 4, block = 'b')            # 1414512
    X = inception_block(X, filters = [128, 128, 256, 24, 64, 64], stage = 4, block = 'c')             # 1414512
    d4 = inception_block(X, filters = [112, 144, 288, 32, 64, 64], stage = 4, block = 'd')            # 1414528
    X = inception_block(d4, filters = [256, 160, 320, 32, 128, 128], stage = 4, block = 'e')          # 1414832
    X = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(X)                                  # 77528
    X = inception_block(X, filters = [256, 160, 320, 32, 128, 128], stage = 5, block = 'a')           # 77832
    X = inception_block(X, filters = [384, 192, 384, 48, 128, 128], stage = 5, block = 'b')           # 771024
    X = AveragePooling2D(pool_size=(7,7), strides=(1,1), padding='valid')(X)                             # 111024 
    X = Flatten()(X)
    X = Dropout(0.4)(X)
    X = Dense(1000, activation='relu')(X)
    final_output = Dense(classes, activation='softmax')(X)
    
    partial_4a_output = AveragePooling2D(pool_size=(5,5), strides=(3,3), padding='valid')(a4)
    partial_4a_output = Conv2D(filters=128, kernel_size=(1,1), strides=(1,1),padding='same', activation='relu')(partial_4a_output)
    partial_4a_output = Dense(1024, activation='relu')(partial_4a_output)
    partial_4a_output = Dropout(0.7)(partial_4a_output)
    partial_4a_output = Dense(classes, activation='softmax')(partial_4a_output)
    
    partial_4d_output = AveragePooling2D(pool_size=(5,5), strides=(3,3), padding='valid')(d4)
    partial_4d_output = Conv2D(filters=128, kernel_size=(1,1), strides=(1,1),padding='same', activation='relu')(partial_4d_output)
    partial_4d_output = Dense(1024, activation='relu')(partial_4d_output)
    partial_4d_output = Dropout(0.7)(partial_4d_output)
    partial_4d_output = Dense(classes, activation='softmax')(partial_4d_output)
    
    
    
    model = Model(inputs= X_input, outputs= [final_output, partial_4a_output, partial_4d_output])
    
    if weight_path:
        model.load_weights(weight_path)
    
    return model
    
    


# In[5]:


model = GoogLeNet(input_shape = (224, 224, 3), classes = 1000)


# In[6]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[7]:


model.summary()


# In[9]:


plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

