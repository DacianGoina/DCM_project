
'''
This class represent a static classifier for a model. Static classifier in this context refer to a classical solver (model) e.g
RandomForestClassifier, XGBoostClassifier, SVM etc; these are heavily different from a Deep Learning model or Neural Network that usually
require a complex architecture and other things that must to be considered.

This class works as a wrapper for classifiers from sklearn package; it incorporate the already implemented features from sklearn
and allow user to use them in a proper way.

'''

from src.main.model_utilities import *

class StaticClassifier:

    def __init__(self, model_classifier):
        '''
        Constructor, where only the classifier object is passed (it does not need to store the data)
        :param model_classifier: the model classifier
        '''
        self.model_classifier = model_classifier
        self.__confusion_matrix = None
        # confusion matrix is a private member (variable); set it after fit_train_evaluate()
        # get it via get_confusion_matrix()

    def set_model_params(self, new_params):
        '''
        Setter for new parameter data
        :param new_params: the new input variable; represented by a map where we have the key - model parameter name and the value - parameter value, e.g 'n_estimators': 200'
        '''
        self.model_classifier.set_params(**new_params)

    def get_model_params(self):
        '''
        Getter for the param data
        :return: a copy of the content of the data param
        '''
        return self.model_classifier.get_params().copy()

    def fit_train_evaluate(self, dict_data):
        '''
        Method for fitting the data, training the model, testing the result, and returning the corresponding metrics of the confusion matrix
        :param dict_data: dictionary with the data (we assume that the given data is preprocessed)
        :return: dictionary with the metrics resulted from the confusion matrix
        :rtype: build-in python dictionary
        '''
        self.model_classifier.fit(dict_data[X_TRAIN], dict_data[Y_TRAIN])
        y_pred = self.model_classifier.predict(dict_data[X_TEST])
        res_conf_matrix = confusion_matrix(dict_data[Y_TEST], y_pred)

        # save confusion matrix into the classifier
        self.__confusion_matrix = res_conf_matrix

        metrics_dict = get_model_evaluation_metrics(res_conf_matrix)
        print("accuracy manually: ", metrics_dict['accuracy'])

        return metrics_dict

    def predict(self, data_point):
        '''
        Method used to predict the label for a value passed as a input (we assume that the given data is preprocessed )
        :param data_point: a given value
        :return: the predicted label
        '''
        predicted_label = self.model_classifier.predict(data_point)
        return predicted_label

    # for a given data_point return the predicted probabilities for every label
    #  assume that the given data is preprocessed
    def predict_probabilities(self, data_point):
        '''
        Method that calculates for a given data_point the predicted probabilities
        :param data_point: a given data point
        :return: list with the predicted probabilities for every label
        :rtype: build-in python list
        '''
        predicted_probabilities = self.model_classifier.predict_proba(data_point)
        predicted_probabilities = predicted_probabilities.flatten().tolist() # convert to built-in list
        predicted_labels_with_probabilities = list(zip(self.get_model_classes(), predicted_probabilities))
        return predicted_labels_with_probabilities

    def get_model_classes(self):
        '''
        Method used to get the model classes
        :return: list with the model classes
        :rtype: build-in python list
        '''
        return list(self.model_classifier.classes_)

    def get_confusion_matrix(self):
        '''
        Method to get the confusion matrix
        :return: list with the confusion matrix
        '''
        return self.__confusion_matrix

    def __str__(self):
        '''
        Method to get the string representation of the class instance
        :return: string with the representations
        '''
        model_parameters = self.get_model_params()
        print_result = self.model_classifier.__class__.__name__ + "\n"
        for param, value in model_parameters.items():
            print_result = print_result + param + ": " + str(value) + "\n"

        print_result = print_result.strip()
        return print_result

    def short_str(self):
        '''
        Name of the class
        :return: string with the name of the class
        :rtype: build-in python string
        '''
        return self.model_classifier.__class__.__name__
