from models.LSTM_model import LSTM_Network,Config
import tensorflow as tf
import numpy as np
tf.compat.v1.disable_eager_execution()

class Build(object):
    def __init__(self,train_data_X,test_data_X,final_train_y,final_test_y) -> None:
        self.train_data_X = train_data_X
        self.test_data_X = test_data_X
        self.final_train_y = final_train_y
        self.final_test_y = final_test_y
    
    def build(self):
        config = Config(self.train_data_X, self.test_data_X)
        # print("Some useful info to get an insight on dataset's shape and normalisation:")
        # print("features shape, labels shape, each features mean, each features standard deviation")
        # print(self.train_data_X.shape, self.final_test_y.shape,
        #         np.mean(self.test_data_X), np.std(self.test_data_X))
        # print("the dataset is therefore properly normalised, as expected.")
        # print(self.train_data_X.shape,  self.final_train_y.shape)

        # Build network
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
        return X, Y, accuracy, cost, optimizer, config, pred_Y

