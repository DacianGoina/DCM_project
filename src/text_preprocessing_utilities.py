# Text preprocessing functions

# The strategy is to use functions without side effects - so do not modify the passes object itself, construct a new way
# that will be returned

import spacy
from num2words import num2words
import re
# nlp_model = spacy.load("en_core_web_sm")


def is_numeric_str(s):
    try:
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

def to_lowercase(text):
    return text.lower()

def remove_excessive_space(text):
    '''
    Remove excessive white spaces like " ", \n, \t from the beginning and ending of text

    :param text - input text; it's a native python string
    :return: the given text without spaces;
    :rtype: built-in python string

    '''
    return text.strip()

# print(remove_excessive_space("\n\n This is a text and this is another one \n \n \t"))


def remove_punctuations(words):
    '''
    Remove all the punctuations from the given text

    :param words: the input list that contains all the words and punctuations
    :return: a list with all words, without punctuations;
    :rtype: built-in python list
    '''

    words_res = []
    for word in words:
        if word.is_punct is False:
            words_res.append(word)

    return words_res


def remove_stopwords(words):
    '''
    Remove all the stop words from the given text

    :param words: the input list that contains all the words
    :return: the given list, without stop words;
    :rtype: built-in python list
    '''
    words_res = []
    for word in words:
        if word.is_stop is False:
            words_res.append(word)

    return words_res

def lemmatize_words(words):
    '''
    Apply lemmatization for a list of words.

    :param words: the input list with words; every element is a spacy.tokens.token.Token object
    :return: a list constructed from the initial one but every with is lemmatized (converted to base form)
    :rtype: built-in python list, every element is a spacy.tokens.token.Token object
    '''

    words_res = []
    for word in words:
        words_res.append(word.lemma_)

    return words_res

def handle_numerical_values(word, method='text'):
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

def handle_rare_words_and_typos(words, threshold=2, replacement='[UNK]'):
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
def words_pos(words):
    words_res = []
    for word in words:
        words_res.append( (word, word.pos_) )

    return words_res

def get_tokens_from_raw_text(text, nlp_model):
    '''
    Convert a raw text to a built-in python list of spacy.tokens.token.Token object (tokens);

    :param text: the input text; it's a native python string
    :param nlp_model: NLP model that is used to preprocess the text; it's a spacy.lang object
    :return: list of words extracted from the input text
    :rtype: built-in python list
    '''
    doc = nlp_model(text)
    words = []
    for token in doc:
        words.append(token)

    return words

def remove_junk_spaces(tokens, nlp_model):

    # convert spacy tokens to str tokens
    tokens = convert_spacy_tokens_to_str_tokens(tokens)

    # remove extra spaces with strip
    tokens = [remove_excessive_space(token) for token in tokens]


    junk_spaces = ['\n', '\t', '\r', '\v', '\f', '&nbsp;', '\xA0', '', ' ']

    # remove other junk spaces
    tokens = [token for token in tokens if token not in junk_spaces]

    tokens = convert_str_tokens_to_spacy_tokens(tokens, nlp_model)

    return tokens



def years_to_spoken_words(tokens, nlp_model):
    # convert years to spoken words, eg. "1990" to 'nineteen ninety'
    # we consider years as integer values with 4 digits, and the value itself
    # between valid_year_min_value to valid_year_max_value

    valid_year_min_value = 1000
    valid_year_max_value = 2100

    # convert tokens from spacy entity to built-in string
    tokens = convert_spacy_tokens_to_str_tokens(tokens)
    new_tokens = []

    for token in tokens:
        if token.isnumeric() and len(token) == 4:
            year = int(token)
            if year >= valid_year_min_value and year <= valid_year_max_value:
                year_as_words = num2words(year, to = 'year')
                new_tokens.append(year_as_words)
            # logica
        else:
            # just append it like this
            new_tokens.append(token)

    new_tokens = convert_str_tokens_to_spacy_tokens(new_tokens, nlp_model)

    return new_tokens

def numeric_values_to_spoken_words(tokens, nlp_model):
    # conver numerical values (eg. '54', '2.5') to spoken words

    new_tokens = []

    # convert tokens from spacy entity to built-in string
    tokens = convert_spacy_tokens_to_str_tokens(tokens)

    for token in tokens:
        if is_numeric_str(token):

            token_as_numeric = float(token)
            token_as_spoken_words = num2words(token_as_numeric)
            new_tokens.append(token_as_spoken_words)
            if token == '0.30':
                print(token_as_numeric)
        else:
            new_tokens.append(token)

    new_tokens = convert_str_tokens_to_spacy_tokens(new_tokens, nlp_model)

    return new_tokens


def convert_str_tokens_to_spacy_tokens(tokens, nlp_model):
    # convert string tokens back into spacy entities
    raw_text = ' '.join(tokens)
    tokens = get_tokens_from_raw_text(raw_text, nlp_model)

    return tokens


def convert_spacy_tokens_to_str_tokens(tokens):
    # convert tokens from spacy entity to built-in string
    tokens = [token.text for token in tokens]
    return tokens

# IN: str tokens
# OUT: str tokens
def convert_ordinal_numbers_to_spoken_words(tokens, nlp_model):
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


# input_text = "The quick brown foxes are jumping over the lazy dogs. They were running through the forests, exploring the mysterious caves. I saw many interesting books on the shelves and decided to read them all. The children were playing happily in the parks, swinging on the swings and climbing on the jungle gym. Despite the challenges, they were determined to succeed in their endeavors."
# words_input = get_tokens_from_raw_text(input_text, nlp_model)
# print("Tokens:")
# print(words_input)
# print("-" * 15)
# words_input = remove_punctuations(words_input)
# print("Without punctuations:")
# print(words_input)
# print("-" * 15)
# words_input = remove_stopwords(words_input)
# print("Without stopwords:")
# print(words_input)
# print("-" * 15)
# words_input = lemmatize_words(words_input)
# print("After lemmatization:")
# print(words_input)
# print("-" * 25)

# # need to convert again to text and then to tokenize because the lemmatization convert words to built in string
# words_as_single_text = ' '.join(words_input)
# words_input = get_tokens_from_raw_text(words_as_single_text, nlp_model)
# words_and_pos = words_pos(words_input)
# print("Part of speech:")
# print(words_and_pos)
# print("-" * 15)
