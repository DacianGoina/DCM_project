
from  FeaturesExtractor import  *
from src.main.model_utilities import *
from sklearn.feature_extraction.text import HashingVectorizer

class HashingVectorizerFE(FeaturesExtractor):
    def __init__(self, data):
        FeaturesExtractor.__init__(self,data)
        self.feature_extractor = HashingVectorizer(lowercase = False, stop_words = None, token_pattern = TOKEN_PATTERN, n_features = 2**15, norm='l2', preprocessor = None, tokenizer = None)
        self.__fit_data(data)

    def set_extractor_params(self, new_params):
        self.feature_extractor.set_params(**new_params)

    def get_extractor_params(self):
        return self.feature_extractor.get_params().copy()

    # should be used only once
    def __fit_data(self, data):
        self.feature_extractor.fit(data)

    def transform_data(self, data):
        result = self.feature_extractor.transform(data)
        super().set_data(data)
        return result

    def get_vocabulary(self):
        return None

    def short_str(self):
        return "HashingVectorizer"