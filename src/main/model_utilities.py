from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from src.main.io_utilities import *
# Constants value to access specific data from @param model_data from class's instances.
X_TRAIN = 'X_train'
X_TEST = 'X_test'
Y_TRAIN = 'y_train'
Y_TEST = 'y_test'


# use the same seed for data split to ensure determinism
SPLIT_DATA_RANDOM_STATE_VALUE = 1

# IN: confusion matrix
# OUT: built-in dict such that keys represent metrics name and data the metrics values
# compute metrics using only confusion matrix, without model itself
# use already implemented functions from sklearn and others modules
# metrics e.g: accuracy, precision, recall, specificity, f1 score, AUC-ROC etc
# TODO
def get_model_evaluation_metrics(confusion_matrix):
    res = dict()

    return res

# IN: df with content and label col
# OUT: X_train, test, Y_train, test
# use stratify=y to split data in a stratified fashion, using this as the class labels: because we have many labels
# we desire a uniform distribution of data with respect to the labels, is not properly if let's labels for training are selected
# data with only 9 distinct labels, and data with tenth label is used only for testing
def split_model_data(X_data = None, y_data = None, test_size_value = 0.25, random_state_val = 0):
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test_size_value, random_state=random_state_val, stratify=y_data)
    return X_train, X_test, y_train, y_test


def build_data_dictionary(X_train, X_test, y_train, y_test):
    data_full_dict = dict()
    data_full_dict[X_TRAIN] = X_train
    data_full_dict[X_TEST] = X_test
    data_full_dict[Y_TRAIN] = y_train
    data_full_dict[Y_TEST] = y_test

    return data_full_dict

# shuffle rows from a pandas df
# in: pandas df
# out: pandas df, rows shuffled by @param no_of_times times
def shuffle_dataframe(df = None, no_of_times = 1):
    new_df = df.copy()
    for i in range(no_of_times):
        new_df = new_df.sample(frac = 1, ignore_index=True)

    return new_df

# IN: dict, key: str value, key: int value
# OUT: None
# save the given dictionary at the given path
def vocabulary_dict_to_json(dictionary, output_file_path):
    save_dict_to_json_file(dictionary, output_file_path)