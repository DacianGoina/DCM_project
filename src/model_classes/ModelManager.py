'''
This class works as a model manager - it create and save model that will be used for the classification.
Manage classifiers and features extractors; along with data split and other steps

The Model Manager flow steps are the following:
    1. received the preprocessed data from our original dataset
    2. create features extractor and classifiers concrete objects
    3. use every features extractor to convert preprocessed text data to numerical features
    4. save features extractor as binary objects, to be reused later
    5. split obtained numerical feature for training and testing stages
    6. use a cross product logic classifiers x features extractors resulting in (classifier, features_extractor) pairs
    7. for every (classifier, features_extractor) pair, the classifier is fitted with the provided data,
        then it is trained and evaluated; the classifiers objects are saved as binary objects to be reused later

Observations:
    1. every features extractors is saved only once (one file per features extractor)
    2. every classifier resulting from any (classifier, features_extractor) is saved, thus for classifiers cl, there will
        k instances of it, one instance for every features extractors that provide data for training / testing for it
    3. classifiers and features extractors to be saved in specific locations


Aim: the persisted model objects: classifiers and features extractor are reused later when a prediction is requested

'''

import sys
sys.path.insert(0, '../src/main')

from src.model_classes.StaticClassifier import StaticClassifier
from src.model_classes.CountVectorizerFE import CountVectorizerFE
from src.model_classes.TfidfVectorizerFE import  TfidfVectorizerFE
from src.model_classes.HashingVectorizerFE import HashingVectorizerFE
from src.model_classes.Doc2VecFE import Doc2VecFE
from src.main.model_utilities import  *
from src.main.io_utilities import *
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.preprocessing import MinMaxScaler
from scipy import sparse

import spacy
import pandas as pd
import itertools
import numpy as np

CLASSIFIERS_OUTPUT_KEY = "classifiers"
EXTRACTORS_OUTPUT_KEY = "extractors"



def manager_execute(preprocessed_data_path, raw_data_path, output_objects_paths, save_model_objs = False):
    '''
    Function to read and process the data set, aggregates all operations of the manager (splitting the data and training the model, can be considered root function)
    :param preprocessed_data_path: the path pf the file; file should be a CSV and we can assume that the data included is preprocessed
    :param raw_data_path: the path pf the file; file should be a CSV
    :param output_objects_paths: dictionary that contains paths for the directories where the resulted objects will be saved
    :param save_model_objs: option if we save or not the objects as binary format
    :return: None
    '''
    # read preprocessed data
    preprocessed_data = pd.read_csv(preprocessed_data_path)

    # read raw data
    raw_data = pd.read_csv(raw_data_path)

    # shuffle processed data
    preprocessed_data = shuffle_dataframe(preprocessed_data, no_of_times=3)

    # shuffle raw data
    raw_data = shuffle_dataframe(raw_data, no_of_times=3)

    # # get independent features and target variable
    # X_data = data['content']
    # y_data = data['label']

    the_extractors = build_features_extractors(preprocessed_data['content'], raw_data['content'])
    the_classifiers = build_classifiers()

    # initialize and save classifiers / features extractors
    model_fit_train_predict(preprocessed_data, raw_data, the_classifiers, the_extractors, output_objects_paths, save_model_objs)


def model_fit_train_predict(preprocessed_data, raw_data, classifiers, features_extractors, output_objects_paths, save_model_objects = False):
    '''
    Function to initialise the model components, it runs and save the extractors and classifiers
    TODO preprocessed_data, raw_data
    :param classifiers: list with the classifiers, StaticClassifier instances
    :param features_extractors: list with the features extractors, FeaturesExtractor instances
    :param output_objects_paths: dictionary that contains paths for the directories where the resulted objects will be saved
    :param save_model_objects:  option if we save or not the objects as binary format
    :return: dictionary with classifiers amd features extractors used and the metrics obtained
    :rtype: build-in python dictionary
    '''
    results = dict()
    numerical_data = dict()# key: name of extractor, key data: resulted data upon transformation

    # transform X data using every feature extractor, store the results
    for extractor in features_extractors:
        X_data = None

        # for doc2vec use raw data
        if extractor.short_str() == 'Doc2Vec':
            X_data = raw_data['content']
        else:
            X_data = preprocessed_data['content']

        transformed_data = extractor.transform_data(X_data.copy())
        numerical_data[extractor.short_str()] = transformed_data
        if save_model_objects is True:
            save_model_component(extractor, extractor.short_str(), output_objects_paths[EXTRACTORS_OUTPUT_KEY])

    y_data = preprocessed_data['label']

    # cross product for classifier and data transformed with features extractors
    classifier_to_extractor = itertools.product(classifiers, list(numerical_data.keys()))

    for (classifier, extractor) in classifier_to_extractor:
        numerical_features = numerical_data[extractor]
        X_train, X_test, y_train, y_test = split_model_data(X_data= numerical_features, y_data= y_data, test_size_value = 0.25, random_state_val = SPLIT_DATA_RANDOM_STATE_VALUE)
        data_dict = build_data_dictionary(X_train, X_test, y_train, y_test)

        working_set_name =  get_classifier_to_extractor_str(classifier.short_str(), extractor)
        print(working_set_name)
        resulted_metrics = classifier.fit_train_evaluate(data_dict)
        if save_model_objects is True:
            save_model_component(classifier, working_set_name, output_objects_paths[CLASSIFIERS_OUTPUT_KEY])

        results[working_set_name] = resulted_metrics

    return results


def build_classifiers():
    '''
    Function that initialise all the classifiers
    :return: list with all used classifiers
    :rtype: build-in python list
    '''
    rf = StaticClassifier(RandomForestClassifier(n_estimators = 150))
    svc_cl = StaticClassifier(svm.SVC(kernel='linear', probability = True, random_state = 3))
    dt = StaticClassifier(DecisionTreeClassifier())
    lr = StaticClassifier(LogisticRegression(max_iter = 250, solver = "liblinear"))
    adaboost_cl = StaticClassifier(AdaBoostClassifier(n_estimators = 150, estimator = RandomForestClassifier(n_estimators = 150)))
    naive_bayes = StaticClassifier(MultinomialNB())

    classifiers = [rf, svc_cl, naive_bayes, dt, lr, adaboost_cl]

    return classifiers


def build_features_extractors(processed_data, raw_data):
    '''
    Function that initialise the features extractors
    :param processed_data: pandas data frame  series (the data is just fitted, not transformed)
    :param raw_data: pandas data frame  series
    :return: list with all used features extractors
    :rtype: build-in python list
    '''
    # cv = CountVectorizerFE(processed_data.copy())
    # tfidf = TfidfVectorizerFE(processed_data.copy())
    # hashing_vec = HashingVectorizerFE(processed_data.copy())
    doc2vec = Doc2VecFE(raw_data.copy())

    #features_extractors = [cv, tfidf, hashing_vec, doc2vec]
    features_extractors = [doc2vec]

    return features_extractors


def get_classifier_to_extractor_str(classifier_name, features_extractor_name):
    '''
    Function that returns the pairs of classifiers and features extractors
    :param classifier_name: string with the name of the classifier
    :param features_extractor_name: string with the name of the features extractor
    :return: string with (classifier, features extractor) pair
    :rtype: build-in python string
    '''
    return classifier_name + "_" + features_extractor_name


def reverse_classifier_to_extractor_str(compound_name):
    '''
    Function that return the components of a compound name (classifier, extractor); reverse engineering for get_classifier_to_extractor_str method
    :param compound_name: string with the pairs (classifier, features extractor)
    :return: tuple with classifier, features extractor names
    '''
    components = compound_name.split("_")
    return (components[0], components[1])


# IN: object itself (python object), object name (str), directory path
# save a model component to binary object
def save_model_component(object, object_name, directory_path):
    '''
    Function that saves a model component to a binary object
    :param object: the object we want to transform
    :param object_name: string with the object name
    :param directory_path: the directory where is located
    :return: None
    '''
    file_path = directory_path + '\\' + object_name
    export_as_binary_obj(object, file_path)

if __name__ == "__main__":
    print("Main")
    preprocessed_data_file_path = '../file_name_v6.csv'
    raw_data_file_path = '../all_contents.csv'
    classifiers_objs_output_dir =  "../../model_objects/classifiers"
    extractors_objs_output_dir = "../../model_objects/features_extractors"


    # root function of manager - this start everything
    manager_execute(preprocessed_data_file_path, raw_data_file_path, {CLASSIFIERS_OUTPUT_KEY:classifiers_objs_output_dir, EXTRACTORS_OUTPUT_KEY:extractors_objs_output_dir}, save_model_objs=True)
