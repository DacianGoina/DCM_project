
from src.model_classes.StaticClassifier import StaticClassifier
from sklearn.naive_bayes import GaussianNB
from src.main.model_utilities import *

class GaussianNBClassifier(StaticClassifier):
    def __init__(self, data):
        StaticClassifier.__init__(self, data, GaussianNB())

    def set_model_data(self, new_data):
        X_train_collection = new_data[X_TRAIN].copy()
        X_test_collection = new_data[X_TEST].copy()

        X_train_collection = self.flat_data(X_train_collection)
        X_test_collection = self.flat_data(X_test_collection)

        new_data[X_TRAIN] = X_train_collection
        new_data[X_TEST] = X_test_collection

        self.model_data = new_data

    def predict(self, data_point):
        data_point_aux = data_point.copy()
        data_point_aux = self.flat_data(data_point)

        predicted_label = self.model_classifier.predict(data_point_aux)
        return predicted_label

    def predict_probabilities(self, data_point):
        data_point_flatten = self.flat_data(data_point.copy())
        return super().predict_probabilities(data_point_flatten)


    # naive bayes work with flat data (1d arrays, not matrices)
    def flat_data(self, data):
        data = data.copy().toarray()
        return data
