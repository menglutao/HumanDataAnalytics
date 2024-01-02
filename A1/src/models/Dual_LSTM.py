
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Dense

class Config(object):
    def __init__(self, X_train, X_test):
        # Input data
        self.train_count = len(X_train)  # number of training series
        self.n_steps = len(X_train[0])  # time steps per series
        self.n_inputs = len(X_train[0][0])  # number of features
        self.n_hidden = 32  # number of neurons in the LSTM layer
        self.n_classes = 3  # number of output classes
        self.learning_rate = 0.0025
        self.batch_size = 1500


def Dual_LSTM(_X,config):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(config.n_steps, config.n_inputs)),  
        MaxPooling1D(pool_size=2),
        LSTM(config.n_hidden, return_sequences=True),  # First LSTM layer
        LSTM(config.n_hidden),                         # Second LSTM layer
        Dropout(0.3),                      # Additional dropout layer
        Dense(100, activation='relu'),
        Dropout(0.3),
        Dense(config.n_classes, activation='softmax')
        ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


