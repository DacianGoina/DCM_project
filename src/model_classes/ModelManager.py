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


# IN: input data file (assume that data is already preprocessed, also file must to be a CSV one);
#  param output_objects_paths is a dict that contains paths for the directories where the resulted objects will be saved

# the function that aggregate all - manage / handle all operations of the manager
# this is the root function of manager
def manager_execute(input_data_path, output_objects_paths, save_model_objs = False):
    # read data
    data = pd.read_csv(input_data_path)

    # shuffle data
    data = shuffle_dataframe(data, no_of_times=3)

    # get independent features and target variable
    X_data = data['content']
    y_data = data['label']

    the_extractors = build_features_extractors(X_data)
    the_classifiers = build_classifiers()

    # initialize and save classifiers / features extractors
    model_fit_train_predict(X_data, y_data, the_classifiers, the_extractors, output_objects_paths, save_model_objs)


# IN: X_data, y_data (assume pandas Series),
# classifiers, features_extractors (assume list with StaticClassifier and FeaturesExtractor instances)
# output_objects_paths is a dict that contains paths for the directories where the resulted objects will be saved

# this function prepare (run and save) the extractors and classifiers
# so it initialize the model components
def model_fit_train_predict(X_data, y_data, classifiers, features_extractors, output_objects_paths, save_model_objects = False):
    results = dict()
    numerical_data = dict()# key: name of extractor, key data: resulted data upon transformation

    # transform X data using every feature extractor, store the results
    for extractor in features_extractors:
        transformed_data = extractor.transform_data(X_data.copy())
        numerical_data[extractor.short_str()] = transformed_data
        if save_model_objects is True:
            save_model_component(extractor, extractor.short_str(), output_objects_paths[EXTRACTORS_OUTPUT_KEY])

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

# OUT: list
# construct list of used classifiers
def build_classifiers():
    rf = StaticClassifier(RandomForestClassifier(n_estimators = 150))
    svc_cl = StaticClassifier(svm.SVC(kernel='linear', probability = True, random_state = 3))
    dt = StaticClassifier(DecisionTreeClassifier())
    lr = StaticClassifier(LogisticRegression(max_iter = 250, solver = "liblinear"))
    adaboost_cl = StaticClassifier(AdaBoostClassifier(n_estimators = 150, estimator = RandomForestClassifier(n_estimators = 150)))
    naive_bayes = StaticClassifier(MultinomialNB())

    classifiers = [rf, svc_cl, naive_bayes, dt, lr, adaboost_cl]
    #classifiers = [rf, svc_cl, dt, lr, adaboost_cl]

    return classifiers

# IN: data (X_data - pandas Series); at this step, the data is just fitted, not transformed - see constructors
# OUT: list
# construct list of used features extractors
def build_features_extractors(data):
    cv = CountVectorizerFE(data.copy())
    tfidf = TfidfVectorizerFE(data.copy())
    hashing_vec = HashingVectorizerFE(data.copy())
    doc2vec = Doc2VecFE(data.copy())

    #features_extractors = [cv, tfidf, hashing_vec, doc2vec]
    features_extractors = [cv, doc2vec]
    return features_extractors

# IN: classifier_name, features_extractor_name, both strings
# OUT: string
# create a string representation for (classifier, features extractor) pair
def get_classifier_to_extractor_str(classifier_name, features_extractor_name):
    return classifier_name + "_" + features_extractor_name

# IN: str
# OUT: tuple with classifier, features extractor names
# reverse engineering for get_classifier_to_extractor_str method
# receive a compound name (classifier, extractor) and return his components
def reverse_classifier_to_extractor_str(compound_name):
    components = compound_name.split("_")
    return (components[0], components[1])


# IN: object itself (python object), object name (str), directory path
# save a model component to binary object
def save_model_component(object, object_name, directory_path):
    file_path = directory_path + '\\' + object_name
    export_as_binary_obj(object, file_path)

if __name__ == "__main__":
    print("Main")
    preprocessed_data_file_path = '../file_name_v6.csv'
    classifiers_objs_output_dir =  "../../model_objects/classifiers"
    extractors_objs_output_dir = "../../model_objects/features_extractors"

    # root function of manager - this start everything
    manager_execute(preprocessed_data_file_path, {CLASSIFIERS_OUTPUT_KEY:classifiers_objs_output_dir, EXTRACTORS_OUTPUT_KEY:extractors_objs_output_dir}, save_model_objs=False)
