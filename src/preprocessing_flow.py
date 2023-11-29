# dummy text classification
from text_preprocessing_utilities import *
# load
nlp_model = spacy.load("en_core_web_sm")

first_doc = df.iloc[0]['content']

# first_doc = remove_excessive_space(first_doc)

# tokens = get_tokens_from_raw_text(first_doc, nlp_model)
# print(first_doc)
# print(tokens)

print(first_doc)

def custom_tokenizer(raw_text, nlp_model):

    # convert to lower case
    #raw_text = to_lowercase(raw_text)

    # remove extra spaces in the first phase
    raw_text = remove_excessive_space(raw_text)

    # get tokens
    tokens = get_tokens_from_raw_text(raw_text, nlp_model)

    # remove junk extra spaces
    tokens = remove_junk_spaces(tokens, nlp_model)

    # handle years value - convert years as numerical value into spoken words
    tokens = years_to_spoken_words(tokens, nlp_model)

    # convert currency symbols into spoken words - MAYBE NOT, just removed them

    # convert articulated date into spoken words (e.g '3rd' -> 'third')

    # convert the left numerical values (int, float) into spoken words
    tokens = numeric_values_to_spoken_words(tokens, nlp_model)

    # remove punctuations
    tokens = remove_punctuations(tokens)

    # remove stop words
    tokens = remove_stopwords(tokens)

    # lemmatization
    tokens = lemmatize_words(tokens)
    # after this, the tokens are not longer spacy.tokens.token.Token, but built-in java string


    return tokens


res_tokens = custom_tokenizer(first_doc, nlp_model)

#res_tokens = np.array(res_tokens)
print(type(res_tokens))
print(res_tokens)


# from num2words import num2words

# print(num2words(1990, to = 'year'))
# print(num2words(1990))

# print(type(num2words(1990, to = 'year')))
# print(num2words(2004 , to = 'year'))
# print(num2words(23, to = 'ordinal'))
# print(num2words(0.30))