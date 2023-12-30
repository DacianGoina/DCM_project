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


def get_model_evaluation_metrics(confusion_matrix):
    '''
    Function that computes manually different metrics (e.g. accuracy, precision, recall, specificity, f1 score etc.) using only the confusion matrix
    :param confusion_matrix: calculated confusion matrix from a model
    :return: dictionary that contains as key the metrics and as value the mean value obtained
    :rtype: build-in python dictionary
    '''
    metric_dict = dict()

    total_instances = np.sum(confusion_matrix)
    true_positives = np.diag(confusion_matrix)

    false_positives = np.sum(confusion_matrix, axis=0) - true_positives
    false_negatives = np.sum(confusion_matrix, axis=1) - true_positives
    true_negatives = total_instances - (true_positives + false_positives + false_negatives)

    accuracy = np.sum(true_positives) / total_instances

    precisions = true_positives / (true_positives + false_positives)
    recalls = true_positives / (true_positives + false_negatives)
    specificitys = true_negatives / (true_negatives + false_positives)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)

    metric_dict['accuracy'] = accuracy
    metric_dict['precision'] = round(np.sum(precisions)/10, 5)
    metric_dict['recall'] = round(np.sum(recalls)/10, 5)
    metric_dict['specificity'] = round(np.sum(specificitys)/10, 5)
    metric_dict['f1_score'] = round(np.sum(f1_scores)/10, 5)

    return metric_dict


def split_model_data(X_data, y_data, test_size_value = 0.25, random_state_val = 0):
    '''
    Function for splitting the data into training and testing sets
    Using stratify=y to split data in a stratified fashion, using this as the class labels: because we have many labels
    The scope being a uniform distribution of data with respect to the labels, is not properly if the labels for training are selected
    Obs: data with only 9 distinct labels, and data with tenth label is used only for testing
    :param X_data: input variable
    :param y_data: target variable
    :param test_size_value: proportion of the dataset that will be used for testing
    :param random_state_val: represents reproducibility, using a certain value results in the data splitting being deterministic
    :return: the training and testing sets
    '''
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test_size_value, random_state=random_state_val, stratify=y_data)
    return X_train, X_test, y_train, y_test


def build_data_dictionary(X_train, X_test, y_train, y_test):
    '''
    Construct a dictionary with the training and testing data
    :param X_train: input variables used for training
    :param X_test: input variables used for testing
    :param y_train: target variable for X_train
    :param y_test: target variable for X_test
    :return: dictionary that has as key the type of data stored and as value the training and testing sets
    :rtype: build-in python dictionary
    '''
    data_full_dict = dict()
    data_full_dict[X_TRAIN] = X_train
    data_full_dict[X_TEST] = X_test
    data_full_dict[Y_TRAIN] = y_train
    data_full_dict[Y_TEST] = y_test

    return data_full_dict


def shuffle_dataframe(df, no_of_times = 1):
    '''
    Function that shuffles x times (value given by @param no_of_times) the rows of a pandas data frame
    :param df: pandas data frame
    :param no_of_times: number of times of shuffling
    :return: the panda dataframe shuffled
    :rtype: pandas.core.frame.DataFrame
    '''
    new_df = df.copy()
    for i in range(no_of_times):
        new_df = new_df.sample(frac = 1, ignore_index=True)

    return new_df


def vocabulary_dict_to_json(dictionary, output_file_path):
    '''
    Saves a given dictionary at the provided path
    :param dictionary: dictionary that contains pair of (key, str value/s), (key, int value/s)
    :param output_file_path: path where the user wants to save the dictionary
    :return: None
    '''
    save_dict_to_json_file(dictionary, output_file_path)


