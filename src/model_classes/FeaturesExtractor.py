
'''
This class represent a feature extractor, i.e a entity that is capable to convert text to numerical values that later are used for classification.
The instance of this class receive the data (X) and the a feature extractor from sklearn module;
the data is transformed to numerical values and returned

This class act as a abstract class; instead of using it directly use the derived classes
'''

# regex token pattern for split the sentence (document) into tokens
# this pattern select every printable character (letters, digits, special characters etc), so without spaces e.g " ", \n, \t and others
TOKEN_PATTERN = "\S+"

class FeaturesExtractor:
    def __init__(self,data):
        self.data = data
        self.feature_extractor = None

    def set_data(self, new_data):
        self.data = new_data

    # return last data used for transformation (not the result of transformation)
    def get_data(self):
        return self.data.copy()

    # pure virtual method; fit data into feature extractor; use this before any transform
    def __fit_data(self, data):
        pass

    # pure virtual method
    # transform passed data; assume that extractor has already fitted with some data
    def transform_data(self, data):
        pass

    # set feature extractor params (with a dictionary); pure virtual method
    def set_extractor_params(self, new_params):
        pass

    # get feature extractor params; pure virtual method
    def get_extractor_params(self):
        return None

    # abstract method; return the vocabulary: the pairs of (token-positions) provided by the extractor after the transformation
    def get_vocabulary(self):
        pass
