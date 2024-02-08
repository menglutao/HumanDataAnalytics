# Dense deep networks


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import Adam
import pandas as pd
import numpy as np

def simple_dense_network(num_classes, num_features,input_dim=128, learning_rate=0.001):
    model = Sequential([

        # First Dense Layer
        Dense(256, activation='relu', input_shape=(num_features*input_dim,), kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.3),

        # Second Dense Layer
        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.3),

        # Third Dense Layer
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.3),

        # Output Layer
        Dense(num_classes, activation='softmax')
    ])

    model.summary()
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model