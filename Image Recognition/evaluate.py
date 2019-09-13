# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
import os
import pickle
import shutil
import sys
import numpy as np
from datetime import datetime
from functools import lru_cache
from itertools import cycle

import tensorflow as tf
from keras import regularizers
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint
from keras.initializers import RandomNormal
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model, model_from_json, load_model
from keras.optimizers import Adam, SGD
from keras.utils import Sequence, plot_model
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input as preprocess_function

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from load_dataset import load_accumulated_info_of_genki4k


def init_model(pretrained_model_file_path):

    _, preprocess_input = MobileNet, preprocess_function
    if os.path.isfile(pretrained_model_file_path):
        print("Loading the Keras model from {} ...".format(pretrained_model_file_path))
        model = load_model(pretrained_model_file_path, compile=False)  # model for inference

    return model, preprocess_input



def read_image_file(image_file_path, input_shape):
    # Read image file from disk
    image_content = cv2.imread(image_file_path)
    # Resize the image
    image_content = cv2.resize(image_content, input_shape[:2][::-1])
    # Convert from BGR to RGB
    image_content = cv2.cvtColor(image_content, cv2.COLOR_BGR2RGB)

    return image_content



def predict_img(image_content, model, preprocess_input):

    # Apply preprocess_input function
    image_content = preprocess_input(image_content)
    image_content = np.array([image_content])
    # Generate predictions
    prediction = model.predict(image_content)
    prediction = np.around(prediction[0][0]) # sigmoid

    return prediction.astype(np.int)

def Confusion_Matrix(y_pred, y_true, labels):

    y_true, y_pred = [labels[i] for i in y_true], [labels[i] for i in y_pred]
    maxtrix = confusion_matrix(y_true, y_pred, labels=labels)
    df_cm = pd.DataFrame(maxtrix, index = [i for i in labels],
                                  columns = [i for i in labels])

    sns.heatmap(df_cm, square=True, annot=True)
    plt.axis([0, 2, 2, 0])
    plt.xlabel("prediction")
    plt.ylabel("ground truth")

    plt.title("confusion matrix")
    plt.show()

def real_time_webcam(model, preprocess_input):
    # Create a camera instance
    cam = cv2.VideoCapture(0)

    # Check if instantiation was successful
    if not cam.isOpened():
        raise Exception("Could not open camera")

    label = ["not smile", "smile"]
    while True:
        isGrab, frame = cam.read()
        if isGrab:
            frame = cv2.flip(frame, 1)
            frame_cpy = cv2.resize(frame, model.input_shape[1:][:2][::-1])

            pred = predict_img(frame_cpy, model, preprocess_input)
            cv2.putText(frame, label[pred], (int(frame.shape[0] / 2), int(frame.shape[1] / 2)), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255, 255, 255), thickness=3)
            # Insert FPS/quit text and show image
            frame = cv2.putText(frame, 'Press q to quit', (440, 20), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 0, 255))

            cv2.imshow('Video feed', frame)

        # q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

def main(_):
    print("Getting hyperparameters ...")
    pretrained_model_file_path = "./output/model.h5"

    print("Loading the annotations of the {} dataset ...".format("genki4k"))
    train_and_test_accumulated_info = load_accumulated_info_of_genki4k()

    print("Using customized cross validation splits ...")
    train_and_test_image_file_path_array, train_and_test_smile_ID_array, _, _, _ = train_and_test_accumulated_info

    # dataset: training num/test num
    # genki4k: 3200/800
    print("Splitting the validation dataset ...")
    train_index_array, test_index_array = train_test_split(np.arange(len(train_and_test_image_file_path_array)), test_size=0.2, random_state=0)
    train_accumulated_info = train_and_test_image_file_path_array[train_index_array], train_and_test_smile_ID_array[train_index_array]
    test_accumulated_info = train_and_test_image_file_path_array[test_index_array], train_and_test_smile_ID_array[test_index_array]

    print("Initiating the model ...")
    model, preprocess_input = init_model(pretrained_model_file_path)

    print("Perform evaluation ...")
    prediction_list = []
    for image_file_path in test_accumulated_info[0]:
        image_content = read_image_file(image_file_path, model.input_shape[1:])
        prediction = predict_img(image_content, model, preprocess_input)
        prediction_list.append(prediction)

    y_pred, y_true = np.array(prediction_list), test_accumulated_info[1]
    accuracy = accuracy_score(y_pred, y_true)
    print(accuracy)

    Confusion_Matrix(y_pred, y_true, labels=["not smile", "smile"])

    print("Perform real time evaluation ...")
    real_time_webcam(model, preprocess_input)

    print("All done!")


if __name__ == "__main__":
    tf.app.run()
