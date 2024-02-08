
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Dense


def simple_LSTM(num_classes,num_features ,window_size = 128,learning_rate=0.001):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(window_size, num_features)),  
        MaxPooling1D(pool_size=2),
        LSTM(32),  # First LSTM layer
        Dropout(0.3),                      # Additional dropout layer
        Dense(100, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
        ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


