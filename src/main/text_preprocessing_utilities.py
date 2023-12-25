# Text preprocessing functions

# The strategy is to use functions without side effects - so do not modify the passes object itself, construct a new way
# that will be returned

import spacy
from num2words import num2words
from validate_email_address import validate_email
from dateutil import parser
from collections import Counter
from consts_values import *
import re
from spacy.lang.en.stop_words import STOP_WORDS
from urllib.parse import urlparse

def is_str_numeric(s):
    '''
        Verifies if a string is a numeric value

        :param s: string
        :return: bool value (True or False)
        :rtype: built-in python bool
    '''
    try:
        if infinity_const in s.lower():
            return False

        # try converting to float
        float_value = float(s)
        return True
    except ValueError:
        try:
            # try converting to int
            int_value = int(s)
            return True
        except ValueError:
            return False


def is_str_valid_date(val):
    '''
        Verifies if a string is a valid date

        :param val: string
        :return: bool value (True or False)
        :rtype: built-in python bool
    '''
    if len(val) < 10:
        return False
    try:
        # try to parse
        parser.parse(val)
        return True
    except ValueError:
        return False


def is_str_fraction(val):
    '''
        Verifies if a string is a valid fraction

        :param val: string
        :return: bool value (True or False)
        :rtype: built-in python bool
    '''
    pattern = re.compile("(?:^|\s)([0-9]+/[0-9]+)(?:\s|$)")
    return bool(pattern.match(val))


# def is_str_fraction(text):
#     pattern = re.compile(r'\b\d+/\d+\b')
#     fractions = re.findall(pattern, text)
#     if len(fractions) == 1:
#         return True
#     return False


def get_lowercase_words_from_str(val):
    '''
        Extract only lowercase words from str (e.g. 'one', 'house' from 'one-house', without '-' or other characters)

        :param val: string
        :return: list with all lower case words
        :rtype: built-in python list
    '''
    words = re.findall('[a-z]+', val)
    return words


def to_lowercase(text):
    '''
        Tranform text string in lowercase

        :param test: string
        :return: lower case string
        :rtype: built-in python string
    '''
    return text.lower()


def remove_excessive_space(text):
    '''
    Remove excessive white spaces like " ", \n, \t from the beginning and ending of text
    :param text - input text; it's a native python string
    :return: the given text without spaces;
    :rtype: built-in python string

    '''
    return text.strip()


def remove_spacy_punctuations(tokens):
    '''
    Remove all the punctuations from the given text

    :param tokens: the input list that contains all the words and punctuations
    :return: a list with all words, without punctuations;
    :rtype: built-in python list
    '''

    tokens = [token for token in tokens if token.is_punct is False]

    return tokens

# IN: list of spacy tokens
# OUT: list of str tokens
def lemmatize_spacy_tokens(tokens):
    '''
    Apply lemmatization for a list of words.

    :param tokens: the input list with words; every element is a spacy.tokens.token.Token object
    :return: a list constructed from the initial one but every with is lemmatized (converted to base form)
    :rtype: built-in python list, every element is a spacy.tokens.token.Token object
    '''

    tokens_lemmas = [token.lemma_ for token in tokens]
    return tokens_lemmas

# IN: list of str tokens
# OUT: list of str tokens
# Remove stopwords; stopwords are fetched from a function defined also in this file
def str_tokens_remove_stopwords(tokens):
    stopwords = get_str_stopwords()
    tokens = [token for token in tokens if token not in stopwords]
    return tokens

def get_str_tokens_freq(tokens):
    '''
   Get the frequency of the tokens given as param

   :param tokens: list of str tokens
   :return: a dictionary that has as key the token and as value the frequency
   :rtype: built-in python dictionary
   '''
    freq = dict()
    freq = {token: tokens.count(token) for token in set(tokens)}
    return freq

# IN: list of lists of tokens: [ [], [], [], ... ]
# OUT: dict, d[token] := frequency of token counting tokens from all lists
def get_str_tokens_freq_for_lists(lists_of_tokens):
    '''
    Get the frequency of a lists of tokens

   :param lists_of_tokens: lists of str tokens ([[], [], ...])
   :return: a dictionary that has as key the token and as value the frequency from all lists
   :rtype: built-in python dictionary
   '''
    main_dict = dict()
    for tokens in lists_of_tokens:
        local_dict = get_str_tokens_freq(tokens)
        main_dict = dict(Counter(main_dict) + Counter(local_dict))

    return main_dict


def get_rare_tokens(dict_of_freq, threshold):
    '''
    Get all values that have a frequency smaller than a threshold

   :param lists_of_tokens: dictionary of frequency
   :param threshold: a value of comparition
   :return: a new dictionary that contains just the frequency's smaller then the threshold
   :rtype: built-in python list
   '''
    res = {token:dict_of_freq[token] for token in dict_of_freq.keys() if dict_of_freq[token] <= threshold}
    return res


def handle_rare_str_tokens(tokens, dict_of_freq, replace_with = rare_token_replacement_tag):
    '''
    Filter tokens by eliminating rare tokens

   :param tokens: list of tokens
   :param dict_of_freq: dictionary of tokens frequency
   :param replace_with: token value for replacement (if None, remove tokens, else replace with given value)
   :return: filtered list of tokens without items in dictionary
   :rtype: built-in python list
   '''
    rare_tokens_list = list(dict_of_freq.keys())
    if replace_with == None: # remove rare tokens
        tokens = [token for token in tokens if token not in rare_tokens_list]
    else: # replace rare tokens with a constant value
        tokens = [token if token not in rare_tokens_list else rare_token_replacement_tag for token in tokens]

    return tokens


def spacy_tokens_pos(tokens):
    '''
    Creates a list of tuples with every token and it's frequency

   :param tokens: list of spacy tokens
   :return: list of tuples that contains the token and it's position
   :rtype: built-in python list
   '''
    res = []
    for token in tokens:
        res.append( (token, token.pos_) )

    return res


def get_spacy_tokens_from_raw_text(text, nlp_model):
    '''
    Convert a raw text to a built-in python list of spacy.tokens.token.Token object (tokens);

    :param text: the input text; it's a native python string
    :param nlp_model: NLP model that is used to preprocess the text; it's a spacy.lang object
    :return: list of words extracted from the input text
    :rtype: built-in python list
    '''
    doc = nlp_model(text)
    tokens = []
    for token in doc:
        tokens.append(token)

    return tokens

def str_remove_junk_spaces(tokens):
    '''
    Removes junk spaces from a list of string tokens

    :param tokens: list of string tokens
    :return: list of tokens without junk spaces
    :rtype: built-in python list
    '''
    # remove extra spaces with strip
    tokens = [remove_excessive_space(token) for token in tokens]

    junk_spaces = ['\n', '\t', '\r', '\v', '\f', '&nbsp;', '\xA0', '', ' ']

    # remove other junk spaces
    tokens = [token for token in tokens if token not in junk_spaces]

    return tokens

# TODO remake tests now it use extend not append
# now, list of tokens are appended to main tokens list
# before: append "year as spoken", now append ["year", "as", "spoken"]
def str_years_to_spoken_words(tokens):
    '''
    Transform the numerical years in their equivalent text
    :param tokens: list of str tokens
    :return: modified list of str tokens that has all numerical year instances transformed in their equivalent text
    :rtype: built-in python list
    '''
    # convert years to spoken words, e.g. "1990" to 'nineteen ninety'
    # we consider years as integer values with 4 digits, and the value itself
    # between valid_year_min_value to valid_year_max_value

    valid_year_min_value = 1000
    valid_year_max_value = 2100

    new_tokens = []

    for token in tokens:
        if token.isnumeric() and len(token) == 4 and int(token) >= valid_year_min_value and int(token) <= valid_year_max_value:
            year = int(token)
            year_as_words = num2words(year, to = 'year')
            year_words_tokens = get_lowercase_words_from_str(year_as_words)
            new_tokens.extend(year_words_tokens)
        else:
            # just append it like this
            new_tokens.append(token)

    return new_tokens

# TODO remake tests now it use extend not append
def str_numeric_values_to_spoken_words(tokens):
    '''
    Transform all numerical values to their equivalent in text
    :param tokens: list of str tokens
    :return: modified list of str tokens that contains all numerical values in text
    :rtype: built-in python list
    '''
    # convert numerical values (eg. '54', '2.5') to spoken words
    new_tokens = []

    for token in tokens:
        if is_str_numeric(token):
            token_as_numeric = float(token)
            token_as_spoken_words = num2words(token_as_numeric)
            tokens_spoken_words = get_lowercase_words_from_str(token_as_spoken_words)
            new_tokens.extend(tokens_spoken_words)
        else:
            new_tokens.append(token)

    return new_tokens

# TODO remake tests now it use extend not append
def str_ordinal_numbers_to_spoken_words(tokens):
    '''
    Transform all ordinal values to their equivalent in text
    :param tokens: list of str tokens
    :return: modified list of str tokens that contains all ordinal values in text
    :rtype: built-in python list
    '''
    new_tokens = []

    ordinals = ['st', 'nd', 'rd', 'th']

    # first phase, handle "1st", "23rd" etc
    for token in tokens:
        if (token[-2:].lower() in ordinals) and is_str_numeric(token[0:-2]):
            numerical_part = token[0:-2]
            numerical_part_spoken_whole_str = num2words(float(token[0:-2]), to = "ordinal")
            numerical_part_spoken = get_lowercase_words_from_str(numerical_part_spoken_whole_str)
            new_tokens.extend(numerical_part_spoken)
        else:
            new_tokens.append(token)

    tokens = new_tokens.copy()
    new_tokens = []

    # second phase of conversion process; e.g "1" "st" => "first"
    i = 0
    while i<= len(tokens) - 2:
        if is_str_numeric(tokens[i]) and tokens[i+1].lower() in ordinals:
            ordinal_as_spoken_words_whole_str = num2words(float(tokens[i]), to = 'ordinal')
            ordinal_as_spoken_words = get_lowercase_words_from_str(ordinal_as_spoken_words_whole_str)
            new_tokens.extend(ordinal_as_spoken_words)
            i = i + 1
        else:
            new_tokens.append(tokens[i])

        i = i + 1

    if len(tokens) >= 2 and tokens[-1].lower() not in ordinals:
        new_tokens.append(tokens[-1])

    tokens = new_tokens.copy()
    new_tokens = []

    # third phase of conversion process; e.g "1 st" => "first"
    for token in tokens:
        token_last2chars = token[len(token) -2:]
        token_first_chars = token[0:-2]
        if token_first_chars.isnumeric() and token_last2chars.lower() in ordinals:
            ordinal_as_spoken_words_whole_str = num2words(float(token_first_chars), to = 'ordinal')
            ordinal_as_spoken_words = get_lowercase_words_from_str(ordinal_as_spoken_word_whole_str)
            new_tokens.extend(ordinal_as_spoken_words)
        else:
            new_tokens.append(token)

    return new_tokens

def str_currency_to_spoken_words(tokens):
    '''
    Transform all currency values to their equivalent in text
    :param tokens: list of str tokens
    :return: modified list of str tokens that contains all currency values in text
    :rtype: built-in python list
    '''
    new_tokens = []

    symbols = {'%':'percentage', '€':'euros', '$':'dollars', 'CHF':'swiss francs', 'USD':'dollars', 'EUR':'euros',
               '£':'sterling', 'GBP':'sterling', 'JPY':'yens', 'AUD':'dollars', 'CAD':'dollars'}
    for token in tokens:
        if token in symbols.keys():
            new_tokens.append(symbols[token])
        else:
            new_tokens.append(token)

    return new_tokens


def str_remove_common_chars(tokens):
    '''
    Remove all common chars such as '\' and '"' from tokens list
    :param tokens: list of str tokens
    :return: modified list of str tokens that excludes common chars
    :rtype: built-in python list
    '''
    common_chars = ['\'','"']
    tokens =[token for token in tokens if token not in common_chars]
    return tokens


def remove_str_tokens_len_less_than_threshold(tokens, threshold_value):
    '''
    Remove all str tokens that have the length smaller than a threshold value
    :param tokens: list of str tokens
    :param threshold_value: a minim value for the length of a token
    :return: modified list of str tokens that excludes tokens of length smaller than threshold
    :rtype: built-in python list
    '''
    tokens = [token for token in tokens if len(token)>= threshold_value]
    return tokens

# TODO remake tests now it use extend not append
def str_fraction_to_spoken_words(tokens):
    '''
    Transform all numerical fraction included in tokens in their equivalent text (this produce some chained tokens, e.g 'one-half' and not 'one' 'half')
    :param tokens: list of str tokens
    :return: modified list of str tokens that includes numerical fractions in text
    :rtype: built-in python list
    '''
    new_tokens = []

    for token in tokens:
        if is_str_fraction(token):
            if token == '1/2':
                value = 'one-half'
                value_split = get_lowercase_words_from_str(value)
                new_tokens.extend(value_split)
            else:
                fraction_parts = token.split('/')
                numerator = int(fraction_parts[0])
                denominator = int(fraction_parts[1])

                # convert to spoken words
                numerator_as_words = num2words(numerator)
                denominator_as_words = num2words(denominator, to = "ordinal")

                # extract only words, without '-' and others
                numerator_split = get_lowercase_words_from_str(numerator_as_words)
                denominator_split= get_lowercase_words_from_str(denominator_as_words)

                new_tokens.extend(numerator_split)
                new_tokens.extend(denominator_split)
        else:
            new_tokens.append(token)

    return new_tokens

# IN: list of str tokens
# OUT: list of str tokens
# replace email addresses with 'email' tag constant value
def str_emails_to_email_tag(tokens):
    '''
    Replaces all email addresses with '[email]' tag constant value
    :param tokens: list of str tokens
    :return: new list of str tokens with instances of '[email]' instead of email address
    :rtype: built-in python list
    '''
    tokens = [token if validate_email(token) is False else email_tag for token in tokens ]
    return tokens

def str_dates_to_date_tag(tokens):
    '''
    Reconstruct the list of str token replacing the valid dates with a specific tag
    :param tokens: list of str tokens
    :return: new list of str tokens with instances of [date] instead of the date
    :rtype: built-in python list
    '''
    tokens = [token if is_str_valid_date(token) is False else calendar_date_tag for token in tokens]
    return tokens

# IN: list of str tokens
# OUT: list of str tokens
# replace a given symbol with a specific tag
# e.g replace quotes (") with a specific tag: quote;
# important mention: structures such as '"cat' will be converted into '[QUOTE] cat'
def str_tokens_replace_symbol_with_tag(tokens, symbol, tag):
    junk_spaces = [' ', '']
    new_tokens = []
    for token in tokens:
        list_of_indexes = [] # indexes o characters where quote appear

        # collect quote indexes
        current_token_len = len(token)
        for i in range(current_token_len):
            if token[i] == symbol:
                list_of_indexes.append(i)

        if len(list_of_indexes) == 0:
            new_tokens.append(token)

        left = None
        right = None
        for index, index_value in  enumerate(list_of_indexes):
            if index == 0:
                left = 0
            else:
                left = list_of_indexes[index - 1] + 1

            right = list_of_indexes[index]

            if left is not None and right is not None:
                single_str = token[left:right]
                if single_str not in junk_spaces:
                    new_tokens.append(single_str)
                new_tokens.append(tag)

        if right is not None and right != len(token) - 1:
            single_str = token[right + 1: len(token)]
            if single_str not in junk_spaces:
                new_tokens.append(single_str)

    return new_tokens

# IN: list of str tokens
# OUT: list of str tokens
# convert numbers that contain separators to spoken words, e.g "10,500,205" to spoken words, append new words to list
# ["ten", "million", "five", "hundred", "thousand", "two", "hundred", "and", "five"]
# TODO now it use extend not append
def str_tokens_numbers_with_separators_to_spoken_words(tokens):
    # left side must be a white space or line start
    # center must be follow the pattern, digits(,digits)+
    # right must be a white space or line end
    # structure such as '(?<=\s)' refer to match but not to include in the result
    pattern = re.compile("((?<=\s)|(?<=^))(\d+((,\d+)+))((?=\s)|(?=$))")
    new_tokens = []
    for token in tokens:
        matches = re.findall(pattern, token)
        if len(matches) == 1:
            numerical_value = str_number_with_separators_to_integer_number(token)
            numerical_value_spoken = num2words(int(numerical_value), to = "cardinal")
            words = get_lowercase_words_from_str(numerical_value_spoken)
            new_tokens.extend(words)
        else:
            new_tokens.append(token)

    return new_tokens

# IN: str representing a number with comma separator, e.g "10,500,205"
# OUT: int value, e.g 10500205
def str_number_with_separators_to_integer_number(val):
    all_digits = [character for character in val if character.isdigit()]
    numerical_value = int(''.join(all_digits))

    return numerical_value

# IN: list of str
# OUT: list of str
# for tokens that are 'compound' words, e.g 'tech,media' (this is a whole token, not 2) split them by the given separator
# and them gather them together: ['tech', 'media']
def split_and_gather_str_tokens_by_separator(tokens, separator):
    new_tokens = []
    for token in tokens:
        token_parts = token.split(separator)
        new_tokens.extend(token_parts)
    return new_tokens

# OUT: list of str tokens that represent stopwords
def get_str_stopwords():
# do not consider this token as stopwords
    NOT_STOPWORDS = ['one','sixty', 'nine', 'six', 'twelve', 'twenty', 'ten', 'hundred', 'third', 'five', 'two', 'three', 'eleven', 'first', 'four', 'forty', 'fifty', 'fifteen', 'eight']
    stopwords = STOP_WORDS.copy()
    res = [stopword for stopword in stopwords if stopword not in NOT_STOPWORDS]
    return res

# TODO make tests
# IN: str value
# OUT: true of false
# match calendar date with 6 digits format, e.g 14.05.93 (refer to a date from year 93); the separator can be ".", "-", "/"
def is_6digits_date(value):
    pattern = re.compile("\d{2}[.|/|-]\d{2}[.|/|-]\d{2}")
    return bool(pattern.match(value))

# TODO make tests
# IN : list of str tokens
# OUT: list of str tokens
# replace 6 digits dates with "c_date" tag
def str_6digits_dates_to_date_tag(tokens):
    tokens = [token if is_6digits_date(token) is False else calendar_date_tag for token in tokens]
    return tokens

# TODO make tests
# IN: str
# OUT: bool
# check valid URL, URI
def is_valid_url(value):
    try:
        result = urlparse(value)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

# TODO make tests
# IN: str
# OUT: bool
# check url in small notation (kind of resource), "ames.arc.nasa.gov"
def is_valid_resource(value):
    pattern = re.compile("[a-z]+[.][a-z]+([.][a-z]+)+")
    return bool(pattern.match(value))

# TODO make tests
# IN: list of str tokens
# OUT: list of str tokens
# replace URLs with url tag
def str_urls_to_url_tag(tokens):
    tokens = [token if (is_valid_url(token) or is_valid_resource(token)) is False else url_tag for token in tokens]
    return tokens

# TODO make tests
# IN: list of str tokens
# OUT: list of str tokens
# replace initial case letters (e.g "a.", "F.") to a specific tag
def str_initial_case_to_tag(tokens):
    def is_initial_case_letter_format(value):
        if len(value) == 2 and value[0].isalpha() and value[1] == ".":
            return True

        return False

    tokens = [token if is_initial_case_letter_format(token) is False else initial_case_letter for token in tokens]
    return tokens

def str_tokens_to_spacy_tokens(tokens, nlp_model):
    '''
    Transform a list of str tokens in a list of the equivalent spacy entities
    :param tokens: list of str tokens
    :param nlp_model: what kind of modifier we want to use
    :return: new list of str tokens that includes spacy tokens
    :rtype: built-in python list
    '''
    # convert string tokens back into spacy entities
    raw_text = ' '.join(tokens)
    tokens = get_spacy_tokens_from_raw_text(raw_text, nlp_model)

    return tokens


def spacy_tokens_to_str_tokens(tokens):
    '''
    Transform a list of spacy tokens in a list of the equivalent str entities
    :param tokens: list of spacy tokens
    :return: new list of spacy tokens
    :rtype: built-in python list
    '''
    # convert tokens from spacy entity to built-in string
    tokens = [token.text for token in tokens]
    return tokens


def str_tokens_to_str(tokens):
    '''
    Transform a list of str token in a string equivalent
    :param tokens: list of str tokens
    :return: string
    :rtype: built-in python string
    '''
    return ' '.join(tokens)