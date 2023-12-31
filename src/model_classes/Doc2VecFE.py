
from src.model_classes.FeaturesExtractor import FeaturesExtractor
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Doc2VecFE(FeaturesExtractor):

    def __init__(self, data):
        '''
        Initialising the class parameter
        :param data: input variable
        '''
        FeaturesExtractor.__init__(self, data)
        data = self.__convert_data_to_str_tokens(data)
        self.feature_extractor = self.__initialize_extractor(data)

    # IN: data - pandas df series, every row contains str tokens joined with ' ' separator,
    # OR list of str values (list with str elements, every str contains str tokens joined with ' ' separator)
    # OUT: list of lists, every sublist contains str tokens - the single str is split using ' ' separator
    def __convert_data_to_str_tokens(self, data):
        if isinstance(data,pd.Series ):
            data = data.tolist()
        new_data = []
        for data_record in data:
            new_data_record = data_record.split(' ')
            new_data.append(new_data_record)

        return new_data

    # IN: pandas series (row:single str) or built-in list with str elements
    # OUT: list with vectors (lists) of numerical values (feature); values are scaled to be positive values
    def transform_data(self, data):
        super().set_data(data.copy())
        data = self.__convert_data_to_str_tokens(data)
        resulted_vectors = []

        for data_record in data:
            resulted_vector = self.feature_extractor.infer_vector(data_record)
            resulted_vectors.append(resulted_vector)

        # if list contains only one element, we need to reshape it in order to use scaler in a proper way
        # otherwise all the data is scaled to zeros - because the way scaler works on one dimensional arrays
        # this fact is related to the differences between shapes like (150, 1) and (1, 150)
        # so convert to column vector type
        one_elem = False
        if len(resulted_vectors) == 1:
            resulted_vectors = np.array(resulted_vectors)
            resulted_vectors = resulted_vectors.reshape(-1, 1)
            one_elem = True

        # scale data to ensure positive values
        scaler = MinMaxScaler()
        resulted_vectors = scaler.fit_transform(resulted_vectors)

        # convert back to row array
        if one_elem is True:
            resulted_vectors = np.transpose(resulted_vectors)

        return resulted_vectors

    # IN: data to be used for training of the extractor at initialization
    # data to be a list of lists; evey sublist to contain str tokens (so not whole str)
    # OUT: trained feature extractor model that is used to convert list of str tokens to list of numerical values
    def __initialize_extractor(self, data):

        NO_OF_EPOCHS = 35
        OUTPUT_VECTOR_SIZE = 150

        tagged_data =[]
        # create list of TaggedDocument object using provided data
        # each TaggedDocument is created using a sublist (a list with str tokens) from the data list and a reference index
        for index, data_record in enumerate(data):
            tagged_record = TaggedDocument(words=data_record, tags=[str(index)])
            tagged_data.append(tagged_record)


        # from Doc2Vec documentation:
        # vector_size : int, optional - Dimensionality of the feature vectors
        # window : int, optional - The maximum distance between the current and predicted word within a sentence.
        # min_count : int, optional - Ignores all words with total frequency lower than this.
        doc2vec_model = Doc2Vec(vector_size=OUTPUT_VECTOR_SIZE, window=5, min_count=1, workers=4, epochs=NO_OF_EPOCHS)
        doc2vec_model.build_vocab(tagged_data)
        doc2vec_model.train(tagged_data, total_examples=len(data), epochs=NO_OF_EPOCHS)

        return doc2vec_model


    def set_extractor_params(self, new_params):
        '''
        For Doc2Vec extractor we cannot set the params in this way. The params are set at initialization.
        If you want to change the params, it is mandatory to retrain the model again - so there would be a new model
        :param new_params: the new input variable
        '''
        pass

    def get_extractor_params(self):
        '''
        Getter for the param data
        :return: a copy of the content of the data param
        '''
        return self.feature_extractor.get_params().copy()

    def short_str(self):
            '''
            Name of the class
            :return: string with the name of the class
            :rtype: build-in python string
            '''
            return "Doc2Vec"
