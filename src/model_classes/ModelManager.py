'''
This class aggregates several classifiers and features extractors; also, it manage the data preprocessing flow, then
data split and other
'''

import sys
sys.path.insert(0, '../src/main')

from StaticClassifier import StaticClassifier
from CountVectorizerFE import CountVectorizerFE
from TfidfVectorizerFE import  TfidfVectorizerFE
from HashingVectorizerFE import HashingVectorizerFE
from src.main.model_utilities import  *
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import spacy
import pandas as pd
import itertools

# the function that aggregate all
def manager_execute(data, classifiers, features_extractors):
    data = shuffle_dataframe(data, no_of_times=3)

    results = dict()
    classifier_to_extractor = itertools.product(classifiers, features_extractors)

    X_data = data['content']
    y_data = data['label']

    for (classifier, extractor) in classifier_to_extractor:
        numerical_features = extractor.transform_data(X_data)
        X_train, X_test, y_train, y_test = split_model_data(X = numerical_features, y = y_data, test_size_value = 0.25, random_state_val = SPLIT_DATA_RANDOM_STATE_VALUE)
        data_dict = build_data_dictionary(X_train, X_test, y_train, y_test)

        working_set_name = get_classifier_to_extractor_str(classifier, extractor)
        print(working_set_name)
        classifier.set_model_data(data_dict)
        resulted_metrics = classifier.fit_train_evaluate()

        results[working_set_name] = resulted_metrics

    return results

def build_classifiers():
    rf = StaticClassifier(None, RandomForestClassifier())
    svc = StaticClassifier(None, svm.SVC(kernel='linear'))
    dt = StaticClassifier(None, DecisionTreeClassifier())
    gbc = StaticClassifier(None, GradientBoostingClassifier())

    classifiers = [rf, svc, dt, gbc]

    return classifiers

def build_features_extractors():
    cv = CountVectorizerFE(None)
    tfidf = TfidfVectorizerFE(None)
    hashing_vec = HashingVectorizerFE(None)

    features_extractors = [cv, tfidf, hashing_vec]
    return features_extractors

def get_classifier_to_extractor_str(classifier, features_extractor):
    return classifier.short_str() + ";" + features_extractor.short_str()


if __name__ == "__main__":
    print("Main")

    data = pd.read_csv('../file_name_v4.csv')
    the_classifiers = build_classifiers()
    the_extractors = build_features_extractors()

    manager_execute(data, the_classifiers, the_extractors)


