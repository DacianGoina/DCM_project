from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import spacy
import pandas as pd
import numpy as np
from preprocessing_flow import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import accuracy_score

# IN: confusion matrix
# OUT: built-in dict such that keys represent metrics name and data the metrics values
# compute metrics using only confusion matrix, without model itself
# use already implemented functions from sklearn and others modules
# metrics e.g: accuracy, precision, recall, specificity, f1 score, AUC-ROC etc
def get_model_evaluation_metrics(confusion_matrix):
    res = dict()

    return res

# IN: df with content and label col
# OUT: X_train, test, Y_train, test
def split_model_data(X = None, y = None, test_size_value = 0.25, random_state_val = 0):
    # X = data['content']
    # y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_value, random_state=random_state_val)
    return X_train, X_test, y_train, y_test


# encode data using bag of words strategy
# IN: X data (as str containing tokens)
# OUT: X data in bag of words vector format
def encode_str_data_bow(X):
    cv = CountVectorizer()
    X_cv = cv.fit_transform(X)
    return X_cv

# effectively training and testing of the data on a giving classifier
# IN: model data and classifier
# OUT: model performance metrics
def train_and_test_model(X_train, X_test, y_train, y_test, classifier):
    classifier.fit(X_train, y_train)

    # print("X_train :",X_train.shape)
    # print("X_test :",X_test.shape)

    y_pred = classifier.predict(X_test)
    print("accuracy score: ",accuracy_score(y_test, y_pred))

    res_confusion_matrix = confusion_matrix(y_test, y_pred)
    results = get_model_evaluation_metrics(res_confusion_matrix)

    return results


def dummy_classification():
    nlp_model = spacy.load("en_core_web_sm")
    data = pd.read_csv('file_name.csv')
    data = data.dropna()
    vectorized_text_data = encode_str_data_bow(data['content'])
    print(vectorized_text_data.shape)

    X_train, X_test, y_train, y_test = split_model_data(X = vectorized_text_data, y = data['label'], test_size_value = 0.25, random_state_val = 0)
    rf = RandomForestClassifier()
    conf_matrix = train_and_test_model(X_train, X_test, y_train, y_test, rf)

    #res_metrics = get_model_evaluation_metrics(conf_matrix)
    res_metrics = None
    return res_metrics


dummy_classification()

# df = read_raw_data('../data')
# data = process_df(df, nlp_model)
# print(data)
# data.to_csv('file_name.csv', index=False, encoding='utf-8')