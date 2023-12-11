
from  FeaturesExtractor import  *
from sklearn.feature_extraction.text import HashingVectorizer

class HashingVectorizerFE(FeaturesExtractor):
    def __init__(self, X_data):
        FeaturesExtractor.__init__(self,X_data)
        self.feature_extractor = HashingVectorizer(lowercase = False, stop_words = None, token_pattern = TOKEN_PATTERN, n_features = 2**16)

    def set_extractor_params(self, new_params):
        self.feature_extractor.set_params(**new_params)

    # get feature extractor params; pure virtual method
    def get_extractor_params(self):
        return self.feature_extractor.get_params().copy()

    def transform_data(self, new_data):
        super().set_new_data_before_transformation(new_data)
        result = self.feature_extractor.fit_transform(self.X_data)
        return result

    def short_str(self):
        return "HashingVectorizer"