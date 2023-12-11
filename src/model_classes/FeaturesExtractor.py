
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
    def __init__(self, X_data_value = None):
        self.X_data = X_data_value
        self.feature_extractor = None

    def set_data(self, new_data):
        self.X_data = new_data

    def get_data(self):
        return self.X_data.copy()

    # allow user to pass direct data for transformation: by default the transform_data method should use self.X_data
    # but to make the method more independently, the data to be transformed can be passed directly
    # if @param new_data is not None, then self.X_data is replaced with this new value
    def set_new_data_before_transformation(self, new_data):
        if new_data is not None:
            self.X_data = new_data

    # this must be a pure virtual method - implement it in concrete classes
    def transform_data(self, new_data = None):
        return None

    # set feature extractor params (with a dictionary); pure virtual method
    def set_extractor_params(self, new_params):
        pass

    # get feature extractor params; pure virtual method
    def get_extractor_params(self):
        return None
