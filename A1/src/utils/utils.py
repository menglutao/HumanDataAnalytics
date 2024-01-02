from models.Dual_LSTM import Dual_LSTM, Config
from models.LSTM_CNN import LSTM_CNN
from models.DeepConvLSTM import deep_conv_lstm_model
from models.DeepConvLSTM2 import deep_conv_lstm_model_2
from models.DeepConvLSTM3 import deep_conv_lstm_model_3
from models.DeepConvLSTM4 import deep_conv_lstm_model_4
from models.simple_CNN import simple_CNN
from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix,f1_score,precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import os
from sklearn.preprocessing import StandardScaler

WALKING = 1
DESCENDING = 2
ASCENDING = 3

activities_list_to_consider = [WALKING, DESCENDING, ASCENDING]


data_directory_path = 'dataset/labeled-raw-accelerometry-data-captured-during-walking-stair-climbing-and-driving-1.0.0/raw_accelerometry_data'
participant_demog_path = 'dataset/labeled-raw-accelerometry-data-captured-during-walking-stair-climbing-and-driving-1.0.0/participant_demog.csv'


def load_person_df_map(activities_list_to_consider):
        paths = Path(data_directory_path).glob("*.csv")
        df_participant_demog = pd.read_csv(participant_demog_path, index_col='subj_id')
        person_df_map = {}
        for i, person_data_file_path in enumerate(paths):
            person_id = os.path.splitext(os.path.basename(person_data_file_path))[0]
            df = pd.read_csv(person_data_file_path)

            # extracting DFs by activity ID
            activities = {}
            for activity in activities_list_to_consider:
                activities[activity] = df[df['activity'] == activity]

            # extracting participant details
            details = df_participant_demog.loc[person_id]

            person_df_map[person_id] = {
                'activities': activities,
                'details': details
            }

        return person_df_map


def preprocess_data(person_df_map, window_size=128, step_size=64):
    scaler = StandardScaler()
    X, y = [], []

    for person_id, data in person_df_map.items():
        for activity_id, activity_df in data['activities'].items():
            activity_df = activity_df[['lw_x', 'lw_y', 'lw_z', 'lh_x', 'lh_y', 'lh_z', 'la_x', 'la_y', 'la_z', 'ra_x', 'ra_y', 'ra_z']]
            activity_df = scaler.fit_transform(activity_df)

            for start in range(0, len(activity_df) - window_size, step_size):
                end = start + window_size
                X.append(activity_df[start:end])
                y.append(activity_id)

    return np.array(X), np.array(y)


def select_model(model_name,train_data_y,config):
    if model_name == "LSTM_CNN":
        print("Now it LSTM_CNN training!!!!")
        return LSTM_CNN(train_data_y)
    elif model_name == "Dual_LSTM":
        print("Now it Dual LSTM: stacked LSTM: training!!!!")
        return Dual_LSTM(train_data_y,config)
    elif model_name == "DeepConvLSTM":
        print("Now it DeepConvLSTM training!!!!")
        return deep_conv_lstm_model(train_data_y)
    elif model_name == "DeepConvLSTM2":
        print("Now it DeepConvLSTM2 training!!!!")
        return deep_conv_lstm_model_2(train_data_y)
    elif model_name == "DeepConvLSTM3":
        print("Now it DeepConvLSTM3 training!!!!")
        return deep_conv_lstm_model_3(train_data_y)
    elif model_name == "DeepConvLSTM4":
        print("Now it DeepConvLSTM4 training!!!!")
        return deep_conv_lstm_model_4(train_data_y)
    elif model_name == "simple_CNN":
        print("Now it simple_CNN training!!!!")
        return simple_CNN(train_data_y)
    else:
        raise ValueError("Unknown model name")

def train_model(model_name,train_data_X,train_data_y_1d_mapped,test_data_X,test_data_y_1d_mapped):
    config = Config(train_data_X,test_data_X)
    model = select_model(model_name,train_data_y_1d_mapped,config=config)

    # early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    history = model.fit(train_data_X, train_data_y_1d_mapped, batch_size = 32,epochs=10, validation_split=0.2)
    # model = load_model('models/activity_recognition_model.h5') # might use later for showing the running code
    
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

