
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
        '''
        Initialising the class parameter
        :param data: input variable
        '''
        self.data = data
        self.feature_extractor = None

    def set_data(self, new_data):
        '''
        Setter for the data
        :param new_data: the new value of the data param
        :return: None
        '''
        self.data = new_data

    # return last data used for transformation (not the result of transformation)
    def get_data(self):
        '''
        Getter for the data
        :return: a copy of the data object
        '''
        return self.data.copy()

    def __fit_data(self, data):
        '''
        Pure virtual method that fit the data into feature extractor; this need to be used before any transformation
        :param data: data stored in the class param
        :return: None
        '''
        pass

    def transform_data(self, data):
        '''
        Pure virtual method, that transform the passed data (we assume that the extractor has been already fitted with some data)
        :param data: passed data
        :return: None
        '''
        pass

    def set_extractor_params(self, new_params):
        '''
        Pure virtual method, setter for the extractor params (with a dictionary)
        :param new_params: dictionary with the new values for the extractor
        :return: None
        '''
        pass


    def get_extractor_params(self):
        '''
        Pure virtual method, used to get feature extractor params
        :return: None
        '''
        return None

    def get_vocabulary(self):
        '''
        Abstract method, return the vocabulary: the pairs of (token-positions) provided by the extractor after the transformation
        :return: None
        '''
        pass
