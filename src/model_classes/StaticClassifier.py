
'''
This class represent a static classifier for a model. Static classifier in this context refer to a classical solver (model) e.g
RandomForestClassifier, XGBoostClassifier, SVM etc; these are heavily different from a Deep Learning model or Neural Network that usually
require a complex architecture and other things that must to be considered.

This class works as a wrapper for classifiers from sklearn package; it incorporate the already implemented features from sklearn
and allow user to use them in a proper way.

'''

from src.main.model_utilities import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.main.model_utilities import *

class StaticClassifier:

    # constructor; pass only classifier object, it do not need to store the data
    def __init__(self, model_classifier):
        self.model_classifier = model_classifier
        self.__confusion_matrix = None
        # confusion matrix is a private member (variable); set it after fit_train_evaluate()
        # get it via get_confusion_matrix()

    # set model params (e.g if you want to change the params after initialization)
    # new params is a map, key - model parameter name, value - parameter value, e.g 'n_estimators': 200'
    def set_model_params(self, new_params):
        self.model_classifier.set_params(**new_params)

    def get_model_params(self):
        return self.model_classifier.get_params().copy()

    # fit model with data, train the model, test the result, and return confusion matrix
    # assume that the given data is preprocessed
    def fit_train_evaluate(self, dict_data):
        self.model_classifier.fit(dict_data[X_TRAIN], dict_data[Y_TRAIN])
        y_pred = self.model_classifier.predict(dict_data[X_TEST])
        res_conf_matrix = confusion_matrix(dict_data[Y_TEST], y_pred)

        # save confusion matrix into the classifier
        self.__confusion_matrix = res_conf_matrix

        metrics_dict = get_model_evaluation_metrics(res_conf_matrix)
        print("accuracy manually: ", metrics_dict['accuracy'])
        print("accuracy automate: ", accuracy_score(y_pred,dict_data[Y_TEST] ))
        #
        # print("precision calculated manually: ", metrics_dict['precision'])
        # print("precision", precision_score(y_pred, dict_data[Y_TEST], average='weighted'))
        #
        # print("recall calculated manually: ", metrics_dict['recall'])
        # print("recall: ", recall_score(y_pred, dict_data[Y_TEST], average='weighted'))
        #
        # print("specificity calculated manually: ", metrics_dict['specificity'])
        #
        # print("f1-score calculated manually: ", metrics_dict['f1_score'])
        # print("f1-score: ", f1_score(y_pred, dict_data[Y_TEST], average='weighted'))

        return metrics_dict

    # predict label for a value passed as a input; assume that the given data is preprocessed
    def predict(self, data_point):
        predicted_label = self.model_classifier.predict(data_point)
        return predicted_label

    # for a given data_point return the predicted probabilities for every label
    #  assume that the given data is preprocessed
    def predict_probabilities(self, data_point):
        predicted_probabilities = self.model_classifier.predict_proba(data_point)
        predicted_probabilities = predicted_probabilities.flatten().tolist() # convert to built-in list
        predicted_labels_with_probabilities = list(zip(self.get_model_classes(), predicted_probabilities))
        return predicted_labels_with_probabilities

    def get_model_classes(self):
        return list(self.model_classifier.classes_)

    # get confusion matrix
    def get_confusion_matrix(self):
        return self.__confusion_matrix

    # string representation for class instance
    def __str__(self):
        model_parameters = self.get_model_params()
        print_result = self.model_classifier.__class__.__name__ + "\n"
        for param, value in model_parameters.items():
            print_result = print_result + param + ": " + str(value) + "\n"

        print_result = print_result.strip()
        return print_result

    # short string representation
    def short_str(self):
        return self.model_classifier.__class__.__name__
