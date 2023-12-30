from data.data_segmentation import DataSegmentation
from data.data_loader import DataLoader
import pandas as pd
from collections import Counter
import tensorflow as tf
from utils.activity_type import ActivityType
from utils.build import Build
from utils.train import Train
from utils.performance import Performance
from models.CNN_LSTM import LSTM_CNN
from models.LSTM_model import LSTM_Network,Config
import matplotlib.pyplot as plt
import numpy as np
import random as rn
import datetime
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix,f1_score,precision_score, recall_score
from keras.callbacks import EarlyStopping


tf.compat.v1.enable_eager_execution()

# tf.compat.v1.disable_eager_execution()

def reset_seeds():
   os.environ["PYTHONHASHSEED"] = "42"
   np.random.seed(42) 
   rn.seed(12345)
   tf.random.set_seed(1234)

def plot(history,cm,labels):
    # plot the model accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['loss'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

def select_model(model_name,train_data_y):
    if model_name == "LSTM_CNN":
        return LSTM_CNN(train_data_y)
    elif model_name == "LSTM_Network":
        return LSTM_Network(train_data_y)
    else:
        raise ValueError("Unknown model name")


def train_model(model_name,train_data_X,train_data_y_1d_mapped,test_data_X,test_data_y_1d_mapped):
    
    model = select_model(model_name,train_data_y_1d_mapped)

    # Test LSTM_CNN
    # model = LSTM_CNN(train_data_y_1d_mapped)
    # early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    history = model.fit(train_data_X, train_data_y_1d_mapped, batch_size = 32,epochs=10, validation_split=0.2)
    # model = load_model('models/activity_recognition_model.h5')
    
    y_prediction = model.predict(test_data_X)
    y_prediction = np.argmax(y_prediction, axis = 1)
    y_test= test_data_y_1d_mapped
    
    result = confusion_matrix(y_test, y_prediction , normalize='pred')
    labels = ["WALKING", "DESCENDING", "ASCENDING"]
    cm = confusion_matrix(y_test, y_prediction)

    plot(history,cm,labels)


    precision = precision_score(y_test, y_prediction, average='weighted')
    recall = recall_score(y_test, y_prediction, average='weighted')
    f1 = f1_score(y_test, y_prediction, average='weighted')

    evaluation_results = model.evaluate(test_data_X, test_data_y_1d_mapped)
    loss, accuracy = evaluation_results
    # Print the accuracy
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return loss,accuracy,precision,recall,f1


def main():


    # Output classes to learn how to classify
    LABELS = [
        "Walking",
        "Descending Stairs",
        "Ascending Stairs"
    ]

    models = [
        LSTM_CNN,
        LSTM_Network
    ]

    data_loader = DataLoader("dataset/labeled-raw-accelerometry-data-captured-during-walking-stair-climbing-and-driving-1.0.0/raw_accelerometry_data")
    data_loader.download_data()
    data_loader.read_files()
   
    data_seg = DataSegmentation(window_duration=2.56, overlap=0.5, sampling_rate=50)


    train_data_X,train_data_y = data_seg(data_loader.train_data)
    test_data_X,test_data_y = data_seg(data_loader.test_data)

    # print("train_data_X.shape:",train_data_X.shape) #train_data_X.shape: (25823, 128, 12)
    # print("train_data_y.shape:",train_data_y.shape) #train_data_y.shape: (25823, 1)

    label_mapping = ActivityType.create_label_mapping()
    # print("label_mapping:",label_mapping) #label_mapping: {1: 0, 2: 1, 3: 2}

    one_hot_encoded_train_y = ActivityType.one_hot(train_data_y, label_mapping)
    one_hot_encoded_test_y = ActivityType.one_hot(test_data_y, label_mapping)

    # print("one_hot_encoded_train_y:",one_hot_encoded_train_y.shape) #one_hot_encoded_train_y: (25823, 1, 3)
    final_train_y = one_hot_encoded_train_y.reshape(one_hot_encoded_train_y.shape[0],-1)
    final_test_y = one_hot_encoded_test_y.reshape(one_hot_encoded_test_y.shape[0],-1)



    

    #'''
    # Test 2 stacked LSTM MODEL
    # builder = Build(train_data_X,test_data_X,final_train_y,final_test_y)
    # X, Y, accuracy, cost, optimizer, config, pred_Y = builder.build()
    # trainer = Train(config,optimizer,pred_Y,cost,accuracy,test_data_X,final_test_y,train_data_X,final_train_y,X,Y)
    # train_losses, train_accuracies, test_losses, test_accuracies = trainer.train()
    # print("test_accuracy:",test_accuracies)
    # print("test_loss:",test_losses)
    
    #'''
    



    
    # Test LSTM_CNN Model    
    print("train_data_X.shape:",train_data_X.shape)
    print("final_train_y.shape:",final_train_y.shape)
    # print("y:",final_train_y)
    
    train_data_y_1d = np.squeeze(train_data_y)
    test_data_y_1d = np.squeeze(test_data_y)

    train_data_y_1d_mapped = np.vectorize(label_mapping.get)(train_data_y_1d)
    test_data_y_1d_mapped = np.vectorize(label_mapping.get)(test_data_y_1d)


    print("1D y:",train_data_y_1d.shape)

    num_runs = 1
    log_file = "training_log17.txt"

    with open(log_file, "w") as file:
        file.write("Training Log\n")
        file.write(f"Date and Time: {datetime.datetime.now()}\n")
        file.write(f"Running the training process {num_runs} times\n\n")
        file.write(f"Seed for this run is : {42}, {12345},{1234} \n\n")
        file.write(f"training epoch: {10} learning rate: {0.001}, batch_size = {32} default setting\n")
        for i in range(num_runs):
            tf.compat.v1.enable_eager_execution()
            reset_seeds() 
            # tf.compat.v1.disable_eager_execution()  # Or enable, depending on your requirement
            model_name = "LSTM_CNN"
            loss,accuracy,precision,recall,f1= train_model(model_name,train_data_X,train_data_y_1d_mapped,test_data_X,test_data_y_1d_mapped)
            file.write(f"Run {i+1}: Accuracy = {accuracy}\n")
            file.write(f"Run {i+1}: Precision = {precision}\n")
            file.write(f"Run {i+1}: Recall = {recall}\n")
            file.write(f"Run {i+1}: F1 = {f1}\n")
            file.write(f"Run {i+1}: Loss = {loss}\n\n")
            tf.keras.backend.clear_session()
    print(f"Training complete. Log written to {log_file}")
    
   
  


  
if __name__ == "__main__":
    main()
