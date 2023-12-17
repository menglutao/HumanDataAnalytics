import tensorflow as tf
import numpy as np
tf.compat.v1.disable_eager_execution()
class Train(object):
    def __init__(self,config,optimizer,pred_Y,cost,accuracy,test_data_X,final_test_y,train_data_X,final_train_y,X,Y) -> None:
        self.config = config
        self.optimizer = optimizer
        self.pred_Y = pred_Y
        self.cost = cost
        self.accuracy = accuracy
        self.test_data_X = test_data_X
        self.final_test_y = final_test_y
        self.train_data_X = train_data_X
        self.final_train_y = final_train_y
        self.X = X
        self.Y = Y

    def train(self):
        test_losses = []
        test_accuracies = []
        train_losses = []
        train_accuracies = []

        # Note that log_device_placement can be turned ON but will cause console spam with RNNs.
        sess = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(log_device_placement=False))
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        best_accuracy = 0.0
        # Start training for each batch and loop epochs
        for i in range(self.config.training_epochs): # 200
            for start, end in zip(range(1, self.config.train_count, self.config.batch_size), #start :size of trainset->99006 , size of batch ->1500
                                range(self.config.batch_size, self.config.train_count + 1, self.config.batch_size)):  # end: from the first batch , end of the dataset. 
                _, loss, acc = sess.run([self.optimizer,  self.cost, self.accuracy], feed_dict={
                    self.X: self.train_data_X[start:end],
                    self.Y: self.final_train_y[start:end]
                    })
                
            train_losses.append(loss)
            train_accuracies.append(acc)

            # Test completely at every epoch: calculate accuracy
            pred_out, accuracy_out, loss_out = sess.run(
                [self.pred_Y, self.accuracy, self.cost],
                feed_dict={
                    self.X: self.test_data_X,
                    self.Y: self.final_test_y
                }
            )
            test_losses.append(loss_out)
            test_accuracies.append(accuracy_out)
            print("traing iter: {},".format(i) +
                " test accuracy : {},".format(accuracy_out) +
                " loss : {}".format(loss_out))
            best_accuracy = max(best_accuracy, accuracy_out)

        print("")
        print("final test accuracy: {}".format(accuracy_out))
        print("best epoch's test accuracy: {}".format(best_accuracy))
        print("")
        return train_losses, train_accuracies,test_losses, test_accuracies
 