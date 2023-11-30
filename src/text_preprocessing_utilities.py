# Text preprocessing functions

# The strategy is to use functions without side effects - so do not modify the passes object itself, construct a new way
# that will be returned

import spacy
from num2words import num2words
from validate_email_address import validate_email
from dateutil import parser
import re


# IN: str
# OUT: True, False: numeric (e.g 45, 4.5) or NOT
def is_str_numeric(s):
    infinity_const = 'infinity'

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

# IN: str
# OUT: True, False depending if the input string represent a valid date or not
def is_str_valid_date(val):
    if len(val) < 10:
        return False
    try:
        # try to parse
        parser.parse(val)
        return True
    except ValueError:
        return False

# IN: str val
# OUT: True, False is val represent a fraction, e.g 1/2, 11/33, 11/32
def is_str_fraction(val):
    pattern = re.compile("(?:^|\s)([0-9]+/[0-9]+)(?:\s|$)")
    return bool(pattern.match(val))

# def is_str_fraction(text):
#     pattern = re.compile(r'\b\d+/\d+\b')
#     fractions = re.findall(pattern, text)
#     if len(fractions) == 1:
#         return True
#     return False


# IN: str
# OUT: list of str tokens
# Extract only lowercase words from str (e.g 'one', 'house', without '-' or other characters)
def get_lowercase_words_from_str(val):
    words = re.findall('[a-z]+', val)
    return words

# IN: str
# OUT: str
def to_lowercase(text):
    return text.lower()

# IN: str
# OUT: str
def remove_excessive_space(text):
    '''
    Remove excessive white spaces like " ", \n, \t from the beginning and ending of text

    :param text - input text; it's a native python string
    :return: the given text without spaces;
    :rtype: built-in python string

    '''
    return text.strip()


# IN: list of spacy tokens
# OUT: list of spacy tokens
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
# OUT: list of spacy tokens
def remove_spacy_stopwords(tokens):
    '''
    Remove all the stop words from the given text

    :param tokens: the input list that contains all the words
    :return: the given list, without stop words;
    :rtype: built-in python list
    '''

    tokens = [token for token in tokens if token.is_stop is False]

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

# MAYBE DELETE THIS
def handle_str_numerical_values(word, method='text'):
    '''
   Decide what to do with numerical values (keep them or remove them)

   :param word:
   :param method:
   '''
    if method == 'text':
        word = re.sub(r'\d+', 'NUM', word)
    elif method == 'remove':
        word = re.sub(r'\d+', '', word)
    return word

# IN: list of str tokens
# OUT: ...
def handle_rare_tokens_and_typos(words, threshold=2, replacement='[UNK]'):
    '''
   Decide if we want to replace rare words or not

   :param text:
   :param threshold:
   :param replacement:
   :return:
   :rtype:
   '''
    word_freq = {word: words.count(word) for word in set(words)}
    rare_words = [word for word, freq in word_freq.items() if freq <= threshold]

    processed_words = [replacement if word in rare_words else word for word in words]

    return processed_words


# part of speech for every word
# IN: list of spacy tokens
# OUT: list of tuples with (token, pos_token)
def spacy_tokens_pos(tokens):
    res = []
    for token in tokens:
        res.append( (token, token.pos_) )

    return res

# IN: str (raw text)
# OUT: list of spacy tokens
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

# IN: list of str tokens
# OUT: list of str tokens
def str_remove_junk_spaces(tokens):
    # remove extra spaces with strip
    tokens = [remove_excessive_space(token) for token in tokens]

    junk_spaces = ['\n', '\t', '\r', '\v', '\f', '&nbsp;', '\xA0', '', ' ']

    # remove other junk spaces
    tokens = [token for token in tokens if token not in junk_spaces]

    return tokens


# IN: list of str tokens
# OUT: list of str tokens
def str_years_to_spoken_words(tokens):
    # convert years to spoken words, eg. "1990" to 'nineteen ninety'
    # we consider years as integer values with 4 digits, and the value itself
    # between valid_year_min_value to valid_year_max_value

    valid_year_min_value = 1000
    valid_year_max_value = 2100

    new_tokens = []

    for token in tokens:
        if token.isnumeric() and len(token) == 4 and int(token) >= valid_year_min_value and int(token) <= valid_year_max_value:
            year = int(token)
            year_as_words = num2words(year, to = 'year')
            new_tokens.append(year_as_words)
        else:
            # just append it like this
            new_tokens.append(token)

    return new_tokens

# IN: list of str tokens
# OUT: list of str tokens
def str_numeric_values_to_spoken_words(tokens):
    # convert numerical values (eg. '54', '2.5') to spoken words
    new_tokens = []

    for token in tokens:
        if is_str_numeric(token):
            token_as_numeric = float(token)
            token_as_spoken_words = num2words(token_as_numeric)
            new_tokens.append(token_as_spoken_words)
        else:
            new_tokens.append(token)

    return new_tokens


# IN: str tokens
# OUT: str tokens
def str_ordinal_numbers_to_spoken_words(tokens):
    new_tokens = []

    ordinals = ['st', 'nd', 'rd', 'th']

    # first phase of conversion process; e.g "1" "st" => "first"
    i = 0
    while i<= len(tokens) - 2:
        if tokens[i].isnumeric() and tokens[i+1].lower() in ordinals:
            ordinal_as_spoken_word = num2words(int(tokens[i]), to = 'ordinal')
            new_tokens.append(ordinal_as_spoken_word)
            i = i + 1
        else:
            new_tokens.append(tokens[i])

        i = i + 1

    # first phase of conversion process; e.g "1 st" => "first"
    new_tokens2 = []
    for token in new_tokens:
        token_last2chars = token[len(token) -2:]
        token_first_chars = token[0:-2]
        if token_first_chars.isnumeric() and token_last2chars.lower() in ordinals:
            ordinal_as_spoken_word = num2words(int(token_first_chars))
            new_tokens2.append(ordinal_as_spoken_word)
        else:
            new_tokens2.append(token)

    return new_tokens2

# IN: list of str tokens
# OUT: list of str tokens
def str_currency_to_spoken_words(tokens):
    new_tokens = []

    symbols = {'%':'percentage', '€':'euros', '$':'dollars', 'CHF':'swiss francs', 'USD':'dollars', 'EUR':'euros',
               '£':'pounds sterling', 'GBP':'pounds sterling', 'JPY':'yens', 'AUD':'dollars', 'CAD':'dollars'}
    for token in tokens:
        if token in symbols.keys():
            new_tokens.append(symbols[token])
        else:
            new_tokens.append(token)

    return new_tokens

# IN: list of str tokens
# OUT: list of str tokens
def str_remove_common_chars(tokens):
    common_chars = ['\'', '"']
    tokens =[token for token in tokens if token not in common_chars]
    return tokens

# IN: list of str tokens
# OUT: list of str token
def remove_str_tokens_len_less_than_threshold(tokens, threshold_value):
    tokens = [token for token in tokens if len(token)>= threshold_value]
    return tokens

# IN: list of str tokens
# OUT: list of str tokens
# OBS: this produce some chained tokens, e.g 'one-half' and not 'one' 'half'
def str_fraction_to_spoken_words(tokens):
    new_tokens = []

    for token in tokens:
        if is_str_fraction(token):
            if token == '1/2':
                value = 'one-half'
                value_splited = get_lowercase_words_from_str(value)
                new_tokens.extend(value_splited)
            else:
                if token == '2rcirc':
                    print('da')

                fraction_parts = token.split('/')
                numerator = int(fraction_parts[0])
                denominator = int(fraction_parts[1])

                # convert to spoken words
                numerator_as_words = num2words(numerator)
                denominator_as_words = num2words(denominator, to = "ordinal")

                # extract only words, without '-' and others
                numerator_splited = get_lowercase_words_from_str(numerator_as_words)
                denominator_splited= get_lowercase_words_from_str(denominator_as_words)

                new_tokens.extend(numerator_splited)
                new_tokens.extend(denominator_splited)
        else:
            new_tokens.append(token)

    return new_tokens

# IN: list of str tokens
# OUT: list of str tokens
# replace email addresses with '[EMAIL]' tag constant value
def str_emails_to_email_tag(tokens):
    email_tag = '[email]'
    tokens = [token if validate_email(token) is False else email_tag for token in tokens ]
    return tokens

def str_dates_to_date_tag(tokens):
    calendar_date_tag = '[c_date]' # calendar date
    tokens = [token if is_str_valid_date(token) is False else calendar_date_tag for token in tokens]
    return tokens

# IN: list of str tokens
# OUT: list of spacy tokens
def str_tokens_to_spacy_tokens(tokens, nlp_model):
    # convert string tokens back into spacy entities
    raw_text = ' '.join(tokens)
    tokens = get_spacy_tokens_from_raw_text(raw_text, nlp_model)

    return tokens

# IN: list of spacy tokens
# OUT: list of str tokens
def spacy_tokens_to_str_tokens(tokens):
    # convert tokens from spacy entity to built-in string
    tokens = [token.text for token in tokens]
    return tokens

# IN: list of str tokens
# OUT: str
def str_tokens_to_str(tokens):
    return ' '.join(tokens)
