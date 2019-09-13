import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
from keras.layers import Dense, GlobalAveragePooling2D, Input, Conv2D
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.utils import Sequence, plot_model
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input as preprocess_function

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from load_dataset import load_accumulated_info_of_genki4k


def init_model(input_shape):

    # Initiate the base model
    model_instantiation, preprocess_input = MobileNet, preprocess_function
    base_model = model_instantiation(input_shape=input_shape, weights="imagenet", include_top=False)

    # Add the subsequent blocks
    global_average_pooling = GlobalAveragePooling2D()(base_model.output)
    classification_target_tensor = Dense(units=1, kernel_initializer=RandomNormal(mean=0.0, stddev=0.001), use_bias=False, activation="sigmoid")(global_average_pooling)# sigmoid

    # Define the  model
    model = Model(inputs=[base_model.input], outputs=[classification_target_tensor])
    # Compile the model
    model.compile(optimizer=Adam(lr=0.00032), loss="binary_crossentropy")# sigmoid
    # Print the summary of the network
    model.summary()

    return model, preprocess_input


def read_image_file(image_file_path, input_shape):
    # Read image file from disk
    image_content = cv2.imread(image_file_path)
    # Resize the image
    image_content = cv2.resize(image_content, input_shape[:2][::-1])
    # Convert from BGR to RGB
    image_content = cv2.cvtColor(image_content, cv2.COLOR_BGR2RGB)

    return image_content

def get_data_generator(accumulated_info, preprocess_input, input_shape):
    """ Data generator yielding tuples (image, label) """
    image_file_path_array, smile_ID_array = accumulated_info

    image_content_list, label_encoding_list = [], []
    batch_size = 64 # 64 images as a batch

    for image_file_path, smile_ID in cycle(zip(image_file_path_array, smile_ID_array)):
        image_content = read_image_file(image_file_path, input_shape)
        image_content_list.append(image_content)
        label_encoding_list.append(smile_ID)# sigmoid

        if len(image_content_list) == batch_size:
            image_content_array, label_encoding_array = np.array(image_content_list, dtype=np.float32), np.array(label_encoding_list, dtype=np.float32)
            # Apply preprocess_input function
            image_content_array = preprocess_input(image_content_array)

            yield image_content_array, label_encoding_array
            image_content_list, label_encoding_list = [], []

class Evaluator(Callback):
    def __init__(self, test_accumulated_info, preprocess_input, input_shape):
        super(Evaluator, self).__init__()

        self.test_image_file_path_array, self.test_smile_ID_array = test_accumulated_info
        self.preprocess_input, self.input_shape = preprocess_input, input_shape

    def image_prediction(self, image_file_path_array):
        image_content_list = []
        for image_file_path in image_file_path_array:
            image_content = read_image_file(image_file_path, self.input_shape)
            image_content_list.append(image_content)
        # Re-instantiate image_content_array
        image_content_array = np.array(image_content_list, dtype=np.float32)
        # Apply preprocess_input function
        image_content_array = self.preprocess_input(image_content_array)
        # Generate predictions
        prediction_array = self.model.predict_on_batch(image_content_array)
        prediction_array = np.around(prediction_array) # sigmoid

        return prediction_array

    def on_epoch_end(self, epoch, logs=None):
        # Extract features
        test_image_prediction = self.image_prediction(self.test_image_file_path_array)
        accuracy = accuracy_score(test_image_prediction, self.test_smile_ID_array)
        # Append the accuracy
        logs["accuracy"] = accuracy


class HistoryLogger(Callback):
    def __init__(self, output_folder_path):
        super(HistoryLogger, self).__init__()

        self.accumulated_logs_dict = {}
        self.output_folder_path = output_folder_path

    def visualize(self, loss_name):
        # Unpack the values
        epoch_to_loss_value_dict = self.accumulated_logs_dict[loss_name]
        epoch_list = sorted(epoch_to_loss_value_dict.keys())
        loss_value_list = [epoch_to_loss_value_dict[epoch] for epoch in epoch_list]

        # Save the figure to disk
        figure = plt.figure()
        if isinstance(loss_value_list[0], dict):
            for metric_name in loss_value_list[0].keys():
                metric_value_list = [loss_value[metric_name] for loss_value in loss_value_list]
                print("{} {} {:.6f}".format(loss_name, metric_name, metric_value_list[-1]))
                plt.plot(epoch_list, metric_value_list, label="{} {:.6f}".format(metric_name, metric_value_list[-1]))
        else:
            plt.plot(epoch_list, loss_value_list, label="{} {:.6f}".format(loss_name, loss_value_list[-1]))
        plt.grid(True)
        plt.legend(loc="best")
        plt.savefig(os.path.join(self.output_folder_path, "{}.png".format(loss_name)))
        plt.close(figure)

    def on_epoch_end(self, epoch, logs=None):  # @UnusedVariable
        # Visualize each figure
        for loss_name, loss_value in logs.items():
            if loss_name not in self.accumulated_logs_dict:
                self.accumulated_logs_dict[loss_name] = {}
            self.accumulated_logs_dict[loss_name][epoch] = loss_value
            self.visualize(loss_name)


def main(_):
    print("Getting hyperparameters ...")
    input_shape = (128, 128, 3)

    output_folder_path = os.path.abspath(os.path.join(__file__, "../output"))
    shutil.rmtree(output_folder_path, ignore_errors=True)
    os.makedirs(output_folder_path)
    print("Recreating the output folder at {} ...".format(output_folder_path))

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
    model, preprocess_input = init_model(input_shape)
    plot_model(model, show_shapes=True, show_layer_names=True, to_file=os.path.join(output_folder_path, "model.png"))

    print("Perform training ...")
    train_generator = get_data_generator(train_accumulated_info, preprocess_input, input_shape)

    test_evaluator_callback = Evaluator(test_accumulated_info=test_accumulated_info, preprocess_input=preprocess_input, input_shape=input_shape)
    modelcheckpoint_callback = ModelCheckpoint(filepath=os.path.join(output_folder_path, "model.h5"), monitor="accuracy", mode="max", save_best_only=True, save_weights_only=False, verbose=1)
    historylogger_callback = HistoryLogger(output_folder_path)

    model.fit_generator(generator=train_generator, steps_per_epoch=50,
                        callbacks=[test_evaluator_callback, modelcheckpoint_callback, historylogger_callback],
                        epochs=50, verbose=2)
    print("All done!")


if __name__ == "__main__":
    tf.app.run()
