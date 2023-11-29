# dummy text classification
from text_preprocessing_utilities import *
from io_utilities import *
import spacy

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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

    # remove tokens with length = 1
    tokens = remove_str_tokens_len_less_than_threshold(tokens, 1)

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

# IN: df with content, label, maybe other cols; content with raw data
# OUT: df with content, label; content is a single str with preprocessed tokens
def process_df(df, nlp_model):
    data = pd.DataFrame(columns=['content','label'])
    data_rows = []
    for content, label, file_path in zip(df['content'], df['type'], df['file_path']):
        print("current file: ", file_path)
        tokens = custom_tokenizer(content, nlp_model)
        tokens_as_single_str = str_tokens_to_str(tokens)
        new_record = pd.DataFrame({'content':tokens_as_single_str, 'label':label}, index = [0])
        data_rows.append(new_record)

    data = pd.concat([data] + data_rows, ignore_index=True)
    return data

# IN: df with content and label col
# OUT: X_train, test, Y_train, test
def split_model_data(X = None, y = None, test_size_value = 0.25, random_state_val = 0):
    # X = data['content']
    # y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_value, random_state=random_state_val)
    return X_train, X_test, y_train, y_test

# prepare model data using bag of words strategy
# IN: X data (as str containing tokens)
# OUT: X data in bag of words vector format
def prepare_model_data_bow(X):
    cv = CountVectorizer()
    X_cv = cv.fit_transform(X)
    return X_cv

# def prepare_model_data_bow(X_train, X_test):
#     cv = CountVectorizer()
#     X_train_cv = cv.fit_transform(X_train)
#     X_test_cv = cv.fit_transform(X_test)
#     return X_train_cv, X_test_cv

# effectively training and testing of the data on a giving classifier
def train_and_test_model(X_train, X_test, y_train, y_test, classifier):
    classifier.fit(X_train, y_train)

    print("X_train :",X_train.shape)
    print("X_test :",X_test.shape)

    y_pred = classifier.predict(X_test)

    accuracy_value = accuracy_score(y_test, y_pred)
    print("accuracy: ", accuracy_value)


nlp_model = spacy.load("en_core_web_sm")

# df = read_raw_data('../data')
# data = process_df(df, nlp_model)
# print(data)
# data.to_csv('file_name.csv', index=False, encoding='utf-8')

data = pd.read_csv('file_name.csv')
data = data.dropna()
print(data['content'])
vectorized_text_data = prepare_model_data_bow(data['content'])
print(vectorized_text_data.shape)

X_train, X_test, y_train, y_test = split_model_data(X = vectorized_text_data, y = data['label'], test_size_value = 0.25, random_state_val = 0)

# print(data.shape)
# X_train, X_test, y_train, y_test = split_model_data(data, 0.25, 1)



# here is a NaN value apparently
# rows_with_nan = data[data.isna().any(axis=1)]
# print(rows_with_nan)

# X_train = prepare_model_data_bow(X_train)
# X_test = prepare_model_data_bow(X_test)

# X_train, X_test = prepare_model_data_bow(X_train, X_test)



rf = RandomForestClassifier()
train_and_test_model(X_train, X_test, y_train, y_test, rf)



# path = 'C:\\Users\\Dacian\\Desktop\\MLO_DCM\\data\\space\\space_17.txt'
# data = read_txt_file(path)
# custom_tokenizer(data['content'],nlp_model)


# #df = read_raw_data('../data')
# file_content = read_txt_file('C:\\Users\\Dacian\\Desktop\\MLO_DCM\\data\\business\\business_10.txt')
# #file_content = read_txt_file('C:\\Users\\Dacian\\Desktop\\MLO_DCM\\data\\space\\space_50.txt')
#
#
# # doc = df.iloc[1]['content']
# doc = file_content['content']
# print(doc)


# tokens = custom_tokenizer(doc, nlp_model)
# print(tokens)

# # Custom transformer using spaCy
# class predictors(TransformerMixin):
#     def transform(self, X, **transform_params):
#         # Cleaning Text
#         return X
#
#     def fit(self, X, y=None, **fit_params):
#         return self
#
#     def get_params(self, deep=True):
#         return {}
#
# l = []
# #bow_vector = CountVectorizer(tokenizer = custom_tokenizer, ngram_range=(1,1))
# bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))
# tfidf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer)
# from sklearn.linear_model import RandomForestClassifier
# classifier = RandomForestClassifier()
# # Create pipeline using Bag of Words
# pipe = Pipeline([
#                  ('vectorizer', bow_vector),
#                  ('classifier', classifier)])
#
# X = df_amazon['verified_reviews'] # the features we want to analyze
# ylabels = df_amazon['feedback'] # the labels, or answers, we want to test against
# X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.3)
#
# # model generation
# pipe.fit(X_train,y_train)
# from sklearn import metrics
# # Predicting with a test dataset
# predicted = pipe.predict(X_test)
# print("accuracy: ",accuracy_score(y_test, predicted))



#bow_vector.fit_transform(l) # fit data into count vector


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