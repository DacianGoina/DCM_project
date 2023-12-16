# dummy text classification
from src.main.text_preprocessing_utilities import *

import spacy
import pandas as pd
from src.main.consts_values import *
from io_utilities import *

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

    # convert tokens such as '"cat' into ['[QUOTE]', 'cat']
    tokens = str_tokens_replace_symbol_with_tag(tokens, symbol = quote_value, tag = QUOTE_TAG)

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

    # convert numbers such as "10,000,000" to spoken words
    tokens = str_tokens_numbers_with_separators_to_spoken_words(tokens)

    # convert token such as "tech,media" into ['tech', 'media']
    # !!! USE THIS AFTER PREPROCESSING OF NUMBERS WITH COMMA SEPARATOR: "10, 000"
    tokens = split_and_gather_str_tokens_by_separator(tokens, separator=",")

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
def process_df(df = None, nlp_model = None):
    data = pd.DataFrame(columns=['content','label', 'path'])
    data_rows = []
    tokens_lists = []
    all_files_path = []

    # obtain and process tokens for every doc
    for content, label, file_path in zip(df['content'], df['type'], df['file_path']):
        print("current file: ", file_path)
        all_files_path.append(file_path)
        tokens = custom_tokenizer(content, nlp_model)
        tokens_lists.append(tokens)

    # get rare tokens for all docs overall
    tokens_freq = get_str_tokens_freq_for_lists(tokens_lists)
    rare_tokens = get_rare_tokens(dict_of_freq=tokens_freq, threshold=2)

    # replace rare tokens with specific tag
    # create rows for new df
    for tokens, label, file_path in zip(tokens_lists, df['type'], all_files_path):
        tokens_copy = tokens.copy()
        tokens_copy = handle_rare_str_tokens(tokens = tokens_copy, dict_of_freq = rare_tokens, replace_with = rare_token_replacement_tag)
        tokens_as_single_str = str_tokens_to_str(tokens_copy)
        new_record = pd.DataFrame({'content':tokens_as_single_str, 'label':label, 'path':file_path}, index = [0])
        data_rows.append(new_record)

    # append new rows to dataset
    data = pd.concat([data] + data_rows, ignore_index=True)
    return data


# read raw data from the given directory (with subdirectories as labels)
# created the initial df, process it with @func process_df and save it (the processed resulted df) into a new csv file
def read_preprocess_and_export(directory_path, output_file_name):
    nlp_model = spacy.load("en_core_web_sm")
    df = read_raw_data(directory_path)
    data = process_df(df, nlp_model)
    data.to_csv(output_file_name, index=False, encoding='utf-8')

# IN: file path (.txt file)
# OUT: str (tokens joined with ' ') the resulted preprocessed content of the file
def preprocess_file(file_path):
    file_content = read_txt_file(file_path)
    nlp_model = spacy.load("en_core_web_sm")
    tokens = custom_tokenizer(file_content['content'], nlp_model)
    result = str_tokens_to_str(tokens)
    return tokens



## TODO
## BE CARE; spacy consider common number as 'four', 'five' as common words;
## to overcome this, we can remove stopwords before converting into spoken words OR use a custom stopwords lists
