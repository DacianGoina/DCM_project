from src.main.text_preprocessing_utilities import *
from src.main.consts_values import *
from src.main.io_utilities import *

import spacy
import pandas as pd


# OUT: a new instances of nlp model, this should be used to provide an nlp model for all cases where it is required
def get_nlp_model():
    nlp_model = spacy.load("en_core_web_sm")
    return nlp_model

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

    # remove common chars
    tokens = str_remove_common_chars(tokens)

    # handle email addresses
    tokens = str_emails_to_email_tag(tokens)

    # handle calendar dates
    tokens = str_dates_to_date_tag(tokens)

    # convert tokens such as '"cat' into ['[QUOTE]', 'cat']
    # tokens = str_tokens_replace_symbol_with_tag(tokens, symbol = quote_value, tag = QUOTE_TAG)

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
    # tokens = split_and_gather_str_tokens_by_separator(tokens, separator=",")

    # replace other symbols as 'USD', '%', '€' etc
    tokens = str_currency_to_spoken_words(tokens)

    # remove stopwords
    tokens = str_tokens_remove_stopwords(tokens)

    # handle 6digits dates
    tokens = str_6digits_dates_to_date_tag(tokens)

    # handle urls
    tokens = str_urls_to_url_tag(tokens)

    # handle initial case letters (e.g surname initial case)
    tokens = str_initial_case_to_tag(tokens)

    # remove tokens with length = 1
    tokens = remove_str_tokens_len_less_than_threshold(tokens, 2)

    # convert str tokens to spacy tokens
    tokens = str_tokens_to_spacy_tokens(tokens, nlp_model)

    # remove punctuations
    tokens = remove_spacy_punctuations(tokens)

    # lemmatization
    tokens = lemmatize_spacy_tokens(tokens)
    # after this, the tokens are not longer spacy.tokens.token.Token, but built-in java string

    return tokens

# IN: list of str tokens, nlp model, number of iterations
# OUT: list of str tokens
# apply the custom tokenizer function iteratively over an raw text given as input (similar to usage of epochs in deep learning)
def apply_custom_tokenizer_iteratively(raw_text, nlp_model, iterations = 2):
    for i in range(iterations):
        tokens = custom_tokenizer(raw_text, nlp_model)
        raw_text = str_tokens_to_str(tokens)

    return tokens

# IN: df with content, label, maybe other cols; content with raw data
# OUT: df with content, label; content is a single str with preprocessed tokens
def process_df(df, nlp_model, preprocessing_iterations = 2):
    data = pd.DataFrame(columns=['content','label', 'path'])
    data_rows = []
    tokens_lists = []
    all_files_path = []

    # obtain and process tokens for every doc
    for content, label, file_path in zip(df['content'], df['type'], df['file_path']):
        print("current file: ", file_path)
        all_files_path.append(file_path)
        tokens = apply_custom_tokenizer_iteratively(content, nlp_model, preprocessing_iterations)
        tokens_lists.append(tokens)

    # get rare tokens for all docs overall
    tokens_freq = get_str_tokens_freq_for_lists(tokens_lists)
    rare_tokens = get_rare_tokens(dict_of_freq=tokens_freq, threshold=2)

    # replace rare tokens with specific tag
    # create rows for new df
    for tokens, label, file_path in zip(tokens_lists, df['type'], all_files_path):
        tokens_copy = tokens.copy()
        tokens_copy = handle_rare_str_tokens(tokens = tokens_copy, dict_of_freq = rare_tokens, replace_with = None)
        tokens_as_single_str = str_tokens_to_str(tokens_copy)
        new_record = pd.DataFrame({'content':tokens_as_single_str, 'label':label, 'path':file_path}, index = [0])
        data_rows.append(new_record)

    # append new rows to dataset
    data = pd.concat([data] + data_rows, ignore_index=True)
    return data


# read raw data from the given directory (with subdirectories as labels)
# created the initial df, process it with @func process_df and save it (the processed resulted df) into a new csv file
def read_preprocess_and_export(directory_path, output_file_name, preprocessing_iterations):
    nlp_model = get_nlp_model()
    df = read_raw_data(directory_path)
    data = process_df(df, nlp_model, preprocessing_iterations)
    data.to_csv(output_file_name, index=False, encoding='utf-8')

# IN: file path (.txt file)
# OUT: str (tokens joined with ' ') the resulted preprocessed content of the file
def preprocess_file(file_path):
    file_content = read_txt_file(file_path)
    nlp_model = get_nlp_model()
    tokens = custom_tokenizer(file_content['content'], nlp_model)
    result = str_tokens_to_str(tokens)
    return tokens

## TODO
## BE CARE; spacy consider common number as 'four', 'five' as common words;
## to overcome this, we can remove stopwords before converting into spoken words OR use a custom stopwords lists
