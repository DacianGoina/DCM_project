# dummy text classification
from text_preprocessing_utilities import *
from io_utilities import *
import spacy

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline

def custom_tokenizer(raw_text, nlp_model):
    # convert to lower case
    raw_text = to_lowercase(raw_text)

    # remove extra spaces in the first phase
    raw_text = remove_excessive_space(raw_text)

    # get spacy tokens
    tokens = get_spacy_tokens_from_raw_text(raw_text, nlp_model)

    # get str tokens to use them for preprocessing
    tokens = spacy_tokens_to_str_tokens(tokens)

    # remove junk extra spaces
    tokens = str_remove_junk_spaces(tokens)

    # handle years value - convert years into spoken words
    tokens = str_years_to_spoken_words(tokens)

    # convert articulated date into spoken words (e.g '3rd' -> 'third')
    tokens = str_ordinal_numbers_to_spoken_words(tokens)

    # convert the left numerical values (int, float) into spoken words
    tokens = str_numeric_values_to_spoken_words(tokens)

    # replace other symbols as 'USD', '%', '€' etc
    tokens = str_symbol_to_spoken_words(tokens)

    # convert str tokens to spacy tokens
    tokens = str_tokens_to_spacy_tokens(tokens, nlp_model)

    # remove punctuations
    tokens = remove_spacy_punctuations(tokens)

    # remove stop words
    tokens = remove_spacy_stopwords(tokens)

    # lemmatization
    tokens = lemmatize_spacy_tokens(tokens)
    # after this, the tokens are not longer spacy.tokens.token.Token, but built-in java string


    return tokens


nlp_model = spacy.load("en_core_web_sm")

#df = read_raw_data('../data')
file_content = read_txt_file('C:\\Users\\Dacian\\Desktop\\MLO_DCM\\data\\business\\business_10.txt')
#file_content = read_txt_file('C:\\Users\\Dacian\\Desktop\\MLO_DCM\\data\\space\\space_50.txt')


# doc = df.iloc[1]['content']
doc = file_content['content']
print(doc)


tokens = custom_tokenizer(doc, nlp_model)
print(tokens)

# Custom transformer using spaCy
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        # Cleaning Text
        return X

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

l = []

#bow_vector = CountVectorizer(tokenizer = custom_tokenizer, ngram_range=(1,1))
bow_vector = CountVectorizer(ngram_range=(1,1))
bow_vector.fit_transform(l) # fit data into count vector


## BE CARE; spacy consider common number as 'four', 'five' as common words;
## to overcome this, we can remove stopwords before converting into spoken words OR use a custom stopwords lists






#
# res_tokens = custom_tokenizer(first_doc, nlp_model)
#
# print(type(res_tokens))
# print(res_tokens)


# from num2words import num2words

# print(num2words(1990, to = 'year'))
# print(num2words(1990))

# print(type(num2words(1990, to = 'year')))
# print(num2words(2004 , to = 'year'))
# print(num2words(23, to = 'ordinal'))
# print(num2words(0.30))