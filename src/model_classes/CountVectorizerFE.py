
from  FeaturesExtractor import  FeaturesExtractor
from sklearn.feature_extraction.text import CountVectorizer

class CountVectorizerFE(FeaturesExtractor):
    def __init__(self, X_data):
        FeaturesExtractor.__init__(self,X_data)
        self.feature_extractor = CountVectorizer(lowercase = False, stop_words = None)

    def set_extractor_params(self, new_params):
        self.feature_extractor.set_params(**new_params)

    # get feature extractor params; pure virtual method
    def get_extractor_params(self):
        return self.feature_extractor.get_params().copy()

    def transform_data(self, new_data):
        super().set_new_data_before_transformation(new_data)
        result = self.feature_extractor.fit_transform(self.X_data)
        return result
