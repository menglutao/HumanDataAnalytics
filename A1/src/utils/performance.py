# accuracy / presicion / recall / F-measure
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from models.LSTM_model import Config




class Performance(object):
    def __init__(self, config: Config):
        # init function can be left empty
        self.batch_size = config.batch_size
        self.training_iters = config.training_epochs * config.train_count
        self.display_iter = self.training_iters // self.batch_size
        

    def plot_performance(self, train_losses, train_accuracies, test_losses, test_accuracies):
        font = {
            'family': 'Bitstream Vera Sans',
            'weight': 'bold',
            'size': 18
        }
        matplotlib.rc('font', **font)

        width, height = 12, 12
        plt.figure(figsize=(width, height))

        indep_train_axis = np.array(range(self.batch_size, (len(train_losses) + 1) * self.batch_size, self.batch_size)) # 20, 21*20,20)
        print("indep_train_axis:",np.shape(indep_train_axis))

        plt.plot(indep_train_axis, np.array(train_losses), "b--", label="Train losses")
        plt.plot(indep_train_axis, np.array(train_accuracies), "g--", label="Train accuracies")

        # indep_test_axis = np.append(
        #     np.array(range(self.batch_size, len(test_losses) * self.display_iter, self.display_iter)[:-1]),
        #     [self.training_iters]
        # )
        # plt.plot(indep_test_axis, np.array(test_losses), "b-", label="Test losses")
        # plt.plot(indep_test_axis, np.array(test_accuracies), "g-", label="Test accuracies")

        plt.title("Training session's progress over iterations")
        plt.legend(loc='upper right', shadow=True)
        plt.ylabel('Training Progress (Loss or Accuracy values)')
        plt.xlabel('Training iteration')

        plt.show()