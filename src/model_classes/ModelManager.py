'''
This class aggregates several classifiers and features extractors; also, it manage the data preprocessing flow, then
data split and other
'''

import sys
sys.path.insert(0, '../src/main')
# sys.path.insert(1, '../preprocessing_flow.py')
# sys.path.insert(2, '../text_preprocessing_utilities.py')

from StaticClassifier import StaticClassifier
from CountVectorizerFE import CountVectorizerFE
from TfidfVectorizerFE import  TfidfVectorizerFE
from sklearn.feature_extraction.text import TfidfTransformer
from src.main.model_utilities import  *
import spacy
import pandas as pd

if __name__ == "__main__":
    print("Main")
    rf_classifier = RandomForestClassifier()
    print(type(rf_classifier))
    sc1 = StaticClassifier(None, rf_classifier)

    nlp_model = spacy.load("en_core_web_sm")
    data = pd.read_csv('../main/file_name_v3.csv')
    data = data.dropna()

    ext_cv = CountVectorizerFE(None)

    vectorized_count_vectorizer = ext_cv.transform_data(data['content'])


    X_train, X_test, y_train, y_test = split_model_data(X = vectorized_count_vectorizer, y = data['label'], test_size_value = 0.25, random_state_val = 0)
    print(X_train.shape)
    print(id(X_train))
    print(X_train)
    #print(X_train.toarray())
    data_dict = build_data_dictionary( X_train, X_test, y_train, y_test)

    sc1.set_model_data(data_dict)
    sc1.fit_train_predict()

    ext_tf = TfidfVectorizerFE(None)
    print(ext_tf.get_extractor_params())
    vectorized_tf = ext_tf.transform_data(data['content'])
    X_train, X_test, y_train, y_test = split_model_data(X = vectorized_tf, y = data['label'], test_size_value = 0.25, random_state_val = 0)
    data_dict = build_data_dictionary( X_train, X_test, y_train, y_test)
    print(id(X_train))
    print(X_train.shape)
    print(X_train)

    sc1.set_model_data(data_dict)
    sc1.fit_train_predict()

