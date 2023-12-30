
from src.model_classes.FeaturesExtractor import  *
from src.main.model_utilities import *
from sklearn.feature_extraction.text import HashingVectorizer

class HashingVectorizerFE(FeaturesExtractor):
    def __init__(self, data):
        '''
        Initialising the class parameter
        :param data: input variable
        '''
        FeaturesExtractor.__init__(self,data)
        self.feature_extractor = HashingVectorizer(lowercase = False, stop_words = None, token_pattern = TOKEN_PATTERN, n_features = 2**15, norm='l2', preprocessor = None, tokenizer = None, alternate_sign = False)
        self.__fit_data(data)

    def set_extractor_params(self, new_params):
        '''
        Setter for new parameter data
        :param new_params: the new input variable
        '''
        self.feature_extractor.set_params(**new_params)

    def get_extractor_params(self):
        '''
        Getter for the param data
        :return: a copy of the content of the data param
        '''
        return self.feature_extractor.get_params().copy()

    def __fit_data(self, data):
        '''
        Method for fitting the data, should be used only once
        :param data: input variable
        :return: None
        '''
        self.feature_extractor.fit(data)

    def transform_data(self, data):
        '''
        Method for transforming the input data
        :param data: input variable
        :return: transformed data
        '''
        result = self.feature_extractor.transform(data)
        super().set_data(data)
        return result

    def get_vocabulary(self):
        '''
        Virtual method for returning the dictionary where keys are unique words in the feature matrix, and values are their corresponding indices
        :return: None
        '''
        return None

    def short_str(self):
        '''
        Name of the class
        :return: string with the name of the class
        :rtype: build-in python string
        '''
        return "HashingVectorizer"