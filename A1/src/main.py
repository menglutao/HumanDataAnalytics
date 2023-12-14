from data.data_segmentation import DataSegmentation
from data.data_loader import DataLoader
from models.LSTM_model import LSTM_Network,Config
from models.CNN_model import read_data,make_input_data,CNNModel
import pandas as pd
from collections import Counter
import tensorflow as tf
from utils.activity_type import ActivityType
import numpy as np
import random as rn
tf.compat.v1.disable_eager_execution()
np.random.seed(42)
rn.seed(12345)
tf.random.set_seed(1234)

def main():

    # Output classes to learn how to classify
    LABELS = [
        "Walking",
        "Descending Stairs",
        "Ascending Stairs"
    ]

    data_loader = DataLoader("dataset/labeled-raw-accelerometry-data-captured-during-walking-stair-climbing-and-driving-1.0.0/raw_accelerometry_data")
    data_loader.download_data()
    data_loader.read_files()
  
    data_seg = DataSegmentation(window_duration=2.56, overlap=0.5, sampling_rate=50)
    train_data_X,train_data_y = data_seg(data_loader.train_data)
    test_data_X,test_data_y = data_seg(data_loader.test_data)

    label_mapping = ActivityType.create_label_mapping()
    print("label_mapping:",label_mapping)
    one_hot_encoded_train_y = ActivityType.one_hot(train_data_y, label_mapping)
    one_hot_encoded_test_y = ActivityType.one_hot(test_data_y, label_mapping)
    final_train_y = one_hot_encoded_train_y.reshape(one_hot_encoded_train_y.shape[0],-1)
    final_test_y = one_hot_encoded_test_y.reshape(one_hot_encoded_test_y.shape[0],-1)

    config = Config(train_data_X, test_data_X)
    print("Some useful info to get an insight on dataset's shape and normalisation:")
    print("features shape, labels shape, each features mean, each features standard deviation")
    print(test_data_X.shape, final_test_y.shape,
          np.mean(test_data_X), np.std(test_data_X))
    print("the dataset is therefore properly normalised, as expected.")
    print(train_data_X.shape, final_train_y.shape)


    # ------------------------------------------------------
    # Step 3: Let's get serious and build the neural network
    # ------------------------------------------------------

    X = tf.compat.v1.placeholder(tf.float32, [None, config.n_steps, config.n_inputs])
    Y = tf.compat.v1.placeholder(tf.float32, [None, config.n_classes])

    pred_Y = LSTM_Network(X, config)

    # # Loss,optimizer,evaluation
    l2 = config.lambda_loss_amount * \
        sum(tf.nn.l2_loss(tf_var) for tf_var in tf.compat.v1.trainable_variables())
    # Softmax loss and L2
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(Y), logits=pred_Y)) + l2
    optimizer = tf.compat.v1.train.AdamOptimizer(
        learning_rate=config.learning_rate).minimize(cost)

    correct_pred = tf.equal(tf.argmax(pred_Y, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

    # --------------------------------------------
    # Step 4: Hooray, now train the neural network
    # --------------------------------------------

    # Note that log_device_placement can be turned ON but will cause console spam with RNNs.
    sess = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(log_device_placement=False))
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    best_accuracy = 0.0
    # Start training for each batch and loop epochs
    for i in range(config.training_epochs):
        for start, end in zip(range(0, config.train_count, config.batch_size),
                              range(config.batch_size, config.train_count + 1, config.batch_size)):
            sess.run(optimizer, feed_dict={X: train_data_X[start:end],
                                           Y: final_train_y[start:end]})

        # Test completely at every epoch: calculate accuracy
        pred_out, accuracy_out, loss_out = sess.run(
            [pred_Y, accuracy, cost],
            feed_dict={
                X: test_data_X,
                Y: final_test_y
            }
        )
        print("traing iter: {},".format(i) +
              " test accuracy : {},".format(accuracy_out) +
              " loss : {}".format(loss_out))
        best_accuracy = max(best_accuracy, accuracy_out)

    print("")
    print("final test accuracy: {}".format(accuracy_out))
    print("best epoch's test accuracy: {}".format(best_accuracy))
    print("")



    
if __name__ == "__main__":
    main()
