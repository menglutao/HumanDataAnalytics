from models.LSTM import simple_LSTM
# from models.LSTM_CNN import LSTM_CNN
from models.DeepConvLSTM import deep_conv_lstm_model
from models.DNN import simple_dense_network
from models.simple_CNN import simple_CNN
from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix,f1_score,precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
data_directory_path = 'dataset/labeled-raw-accelerometry-data-captured-during-walking-stair-climbing-and-driving-1.0.0/raw_accelerometry_data'
participant_demog_path = 'dataset/labeled-raw-accelerometry-data-captured-during-walking-stair-climbing-and-driving-1.0.0/participant_demog.csv'

# global variables
position_flag = True
num_features = 3 if position_flag else 12 # if position_flag is true, then we will only consider the one position data which is 3 features

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
    positions = ['lw']
    

    for person_id, data in person_df_map.items():
        for activity_id, activity_df in data['activities'].items():
            if position_flag: # if position_flag is true, then we will only consider the position data
                for position in positions:
                    activity_df = activity_df[[f'{position}_x', f'{position}_y', f'{position}_z']]
            else: # esle we will consider all the position data
                activity_df = activity_df[['lw_x', 'lw_y', 'lw_z', 'lh_x', 'lh_y', 'lh_z', 'la_x', 'la_y', 'la_z', 'ra_x', 'ra_y', 'ra_z']]
            if activity_df.shape[0] == 0: # which means zero rows
                print(f"Skipping person {person_id} activity {activity_id} due to empty dataframe")
                continue
            activity_df = scaler.fit_transform(activity_df)

            for start in range(0, len(activity_df) - window_size, step_size):
                end = start + window_size
                X.append(activity_df[start:end])
                y.append(activity_id)

    return np.array(X), np.array(y)

def preprocess_data_2(person_df_map, window_size=128, step_size=64):
    scaler = StandardScaler()
    X, y = [], []
    window_size = 128
    step_size = 64
    random_seed = 42  # Set a seed for reproducibility

    # Extract unique person_ids
    all_person_ids = list(person_df_map.keys())

    # Choose 20 random subjects for training
    train_person_ids, test_person_ids = train_test_split(all_person_ids, test_size=0.375, random_state=random_seed)
    
    for person_id, data in person_df_map.items():
        for activity_id, activity_df in data['activities'].items():
            activity_df = activity_df[['lw_x', 'lw_y', 'lw_z', 'lh_x', 'lh_y', 'lh_z', 'la_x', 'la_y', 'la_z', 'ra_x', 'ra_y', 'ra_z']]

            if activity_df.shape[0] == 0:
                print(f"Skipping person {person_id} activity {activity_id} due to an empty dataframe")
                continue

            activity_df = scaler.fit_transform(activity_df)

            for start in range(0, len(activity_df) - window_size, step_size):
                end = start + window_size
                X.append(activity_df[start:end])
                y.append(activity_id)

    # Split data into training and test sets based on selected subjects
    X_train, X_test, y_train, y_test = [], [], [], []

    for i in range(len(X)):
        person_id = all_person_ids[i % len(all_person_ids)]  # Cyclically assign person_id to data points
        if person_id in train_person_ids:
            X_train.append(X[i])
            y_train.append(y[i])
        else:
            X_test.append(X[i])
            y_test.append(y[i])

    # Convert lists to numpy arrays for compatibility with machine learning models
    X_train, X_test, y_train, y_test = (
        np.array(X_train),
        np.array(X_test),
        np.array(y_train),
        np.array(y_test)
    )
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)
    return X_train, y_train,X_test,y_test

def select_model(model_name):
    num_classes = 4
    if model_name == "deep_conv_lstm_model":
        print("Now it LSTM_CNN training!!!!")
        return deep_conv_lstm_model(num_classes,num_features)
    elif model_name == "LSTM":
        print("Now it  LSTM: stacked LSTM: training!!!!")
        return simple_LSTM(num_classes,num_features)
    elif model_name == "simple_dense_network":
        print("Now it simple_dense_network training!!!!")
        return simple_dense_network(num_classes,num_features)
    elif model_name == "simple_CNN":
        print("Now it simple_CNN training!!!!")
        return simple_CNN(num_classes,num_features)
    else:
        raise ValueError("Unknown model name")

def train_model(model_name,X_train,y_train,X_test,y_test):
    if model_name == "simple_dense_network":
        X_train = X_train.reshape((X_train.shape[0], -1))  # Reshaping to 2D for dense network
        X_test = X_test.reshape((X_test.shape[0], -1))

    model = select_model(model_name)


    print(X_train.shape)
    print(y_train.shape)
    

    history = model.fit(X_train, y_train, batch_size = 32,epochs=5, validation_split=0.2)
    # model = load_model('models/activity_recognition_model.h5') # might use later for showing the running code
    
    y_prediction = model.predict(X_test)
    y_prediction = np.argmax(y_prediction, axis = 1)

    result = confusion_matrix(y_test, y_prediction , normalize='pred')
    labels = ["WALKING", "DESCENDING", "ASCENDING","DRIVING"]
    cm = confusion_matrix(y_test, y_prediction)

    # plot(history,cm,labels,model_names=model_name)


    precision = precision_score(y_test, y_prediction, average='weighted')
    recall = recall_score(y_test, y_prediction, average='weighted')
    f1 = f1_score(y_test, y_prediction, average='weighted')

    evaluation_results = model.evaluate(X_test, y_test)
    #save trained model for later use
    print("--------------Model saving -------",model_name)
    # model.save(f'trained_models/{model_name}_model.h5')
    loss, accuracy = evaluation_results
    return loss,accuracy,precision,recall,f1

def plot(history,cm,labels,model_names):

    # plot the model accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['loss'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation','Loss'], loc='upper left')

    plt.savefig(f"{model_names}_accuracy_loss.png")
    # plt.show()

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    # set the font size of the plot
    
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{model_names} Confusion Matrix")
    plt.savefig(f"plots/{model_names}_confusion_matrix.png")
    # plt.show()

