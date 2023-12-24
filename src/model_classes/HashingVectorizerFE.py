
from  FeaturesExtractor import  *
from src.main.model_utilities import *
from sklearn.feature_extraction.text import HashingVectorizer

class HashingVectorizerFE(FeaturesExtractor):
    def __init__(self, X_data):
        FeaturesExtractor.__init__(self,X_data)
        self.feature_extractor = HashingVectorizer(lowercase = False, stop_words = None, token_pattern = TOKEN_PATTERN, n_features = 2**15, norm='l2')

    def set_extractor_params(self, new_params):
        self.feature_extractor.set_params(**new_params)

    # get feature extractor params; pure virtual method
    def get_extractor_params(self):
        return self.feature_extractor.get_params().copy()

    def transform_data(self, new_data, save_features_vocabulary = False, vocabulary_path = None):
        super().set_new_data_before_transformation(new_data)
        result = self.feature_extractor.fit_transform(self.X_data)

        # OBS: HashingVectorizer is a stateless features extractor, it retain / store nothing.
        # Thus, it do not has an vocabulary and I do not have what to save (keep).
        # For an input, the input is everytime mapped to a fixed size array; the size if provided by n_features param.
        # As other models, the models that use HashingVectorizerFE are saved and then reused for prediction;
        # before the prediction, the input document is converted using the HashingVectorizerFE and the output will
        # have the same size as records from training data.

        return result

    def get_vocabulary(self):
        return self.feature_extractor.vocabulary_

    def short_str(self):
        return "HashingVectorizer"