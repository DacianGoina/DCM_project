# dummy text classification
from text_preprocessing_utilities import *
from io_utilities import *

import spacy
import pandas as pd


def custom_tokenizer(raw_text = None, nlp_model = None, consider_numbers_as_stopwords = True):
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

    # remove common chars
    tokens = str_remove_common_chars(tokens)

    # handle email addresses
    tokens = str_emails_to_email_tag(tokens)

    # handle calendaristic dates
    tokens = str_dates_to_date_tag(tokens)

    # remove stopwords before start preprocessing numbers
    # in this way, we'll keep tokens like 'three', 'four'
    # AND also keep some stopwords, e.g in '2004' -> 'two thousand and four' keep the end
    if consider_numbers_as_stopwords == False:
        tokens = str_tokens_to_spacy_tokens(tokens, nlp_model)
        # remove stop words
        tokens = remove_spacy_stopwords(tokens)

        tokens = spacy_tokens_to_str_tokens(tokens)

    # handle years value - convert years into spoken words
    tokens = str_years_to_spoken_words(tokens)

    # convert articulated date into spoken words (e.g '3rd' -> 'third')
    tokens = str_ordinal_numbers_to_spoken_words(tokens)

    # convert the left numerical values (int, float) into spoken words
    tokens = str_numeric_values_to_spoken_words(tokens)

    # convert fractions to spoken words
    tokens = str_fraction_to_spoken_words(tokens)

    # replace other symbols as 'USD', '%', '€' etc
    tokens = str_currency_to_spoken_words(tokens)

    # remove tokens with length = 1
    tokens = remove_str_tokens_len_less_than_threshold(tokens, 2)

    # convert str tokens to spacy tokens
    tokens = str_tokens_to_spacy_tokens(tokens, nlp_model)

    # remove punctuations
    tokens = remove_spacy_punctuations(tokens)

    if consider_numbers_as_stopwords == True:
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


# nlp_model = spacy.load("en_core_web_sm")
# path = 'C:\\Users\\Dacian\\Desktop\\MLO_DCM\\data\\food\\food_14.txt'
# data = read_txt_file(path)
# res = custom_tokenizer(data['content'],nlp_model, consider_numbers_as_stopwords=False)
# print(res)




# str1 = 'the 43/12  of the 7876/323  and hgfh67/dss'
# print(num2words(323, to="ordinal"))
# tokens = str1.split(' ')
# tokens = str_fraction_to_spoken_words(tokens)
# print(tokens)

# import re
# def dummy_f(val):
#     return bool(re.search("\b\d+/\d+\b", val))

# put space or beginning line and white space and line end

# print(dummy_f('1/2Rcirc'))
# print(dummy_f('5431/2aaa'))
# print(dummy_f('1432/24'))
# print(dummy_f(' a12/44g '))


# print(is_str_fraction('93/04/01'))
# print(is_str_fraction('1/2Rcirc'))
# print(is_str_fraction('132/243423'))
# print(is_str_fraction('fds1432/24fdsf'))
# print(is_str_fraction(' a12/44g '))

# inp = "11/423"
# print(num2words(423, to="ordinal"))
# print(num2words(423))
# print(is_str_fraction(inp))
# res =inp.split('/')
# print(res)
#
# i  = 'four hundred and twenty-three'
# import re
# tt = re.findall('[a-z]+', i)
#
# print(tt.extend([1,11]))
# print(tt)




# #df = read_raw_data('../data')
# file_content = read_txt_file('C:\\Users\\Dacian\\Desktop\\MLO_DCM\\data\\business\\business_10.txt')
# #file_content = read_txt_file('C:\\Users\\Dacian\\Desktop\\MLO_DCM\\data\\space\\space_50.txt')
#
#
# # doc = df.iloc[1]['content']
# doc = file_content['content']
# print(doc)



## BE CARE; spacy consider common number as 'four', 'five' as common words;
## to overcome this, we can remove stopwords before converting into spoken words OR use a custom stopwords lists



# from num2words import num2words

# print(num2words(1990, to = 'year'))
# print(num2words(1990))

# print(type(num2words(1990, to = 'year')))
# print(num2words(2004 , to = 'year'))
# print(num2words(23, to = 'ordinal'))
# print(num2words(0.30))