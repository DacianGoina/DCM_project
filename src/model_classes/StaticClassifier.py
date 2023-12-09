
'''
This class represent a static classifier for a model. Static classifier in this context refer to a classical solver (model) e.g
RandomForestClassifier, XGBoostClassifier, SVM etc; these are heavily different from a Deep Learning model or Neural Network that usually
require a complex architecture and other things that must to be considered.

This class works as a wrapper for classifiers from sklearn package; it incorporate the already implemented features from sklearn
and allow user to use them in a proper way.

'''

from src.main.model_utilities import *
from sklearn.metrics import accuracy_score

class StaticClassifier:

    # constructor; assume data from model data is already preprocessed (numerical values not text)
    def __init__(self, model_data, model_classifier):
        self.model_data = model_data
        self.model_classifier = model_classifier

    # set model params (e.g if you want to change the params after initialization)
    # new params is a map, key - model parameter name, value - parameter value, e.g 'n_estimators': 200'
    def set_model_params(self, new_params):
        self.model_classifier.set_params(**new_params)

    def get_model_params(self):
        return self.model_classifier.get_params().copy()

    # train the model, test the result, and return confusion matrix
    def fit_train_predict(self):
        self.model_classifier.fit(self.model_data[X_TRAIN], self.model_data[Y_TRAIN])
        y_pred = self.model_classifier.predict(self.model_data[X_TEST])
        res_conf_matrix = confusion_matrix(self.model_data[Y_TEST], y_pred)

        print(accuracy_score(y_pred,self.model_data[Y_TEST] ))

        return get_model_evaluation_metrics(res_conf_matrix)

    # predict label for a value passed as a input
    def predict(self, data_point):
        predicted_label = self.model_classifier.predict(data_point)
        return predicted_label

    def set_model_data(self, new_data):
        self.model_data = new_data

    def get_model_data(self):
        return self.model_data.copy()

    # string representation for class instance
    def __str__(self):
        model_parameters = self.get_model_params()
        print_result = self.model_classifier.__class__.__name__ + "\n"
        for param, value in model_parameters.items():
            print_result = print_result + param + ": " + str(value) + "\n"

        print_result = print_result.strip()
        return print_result
