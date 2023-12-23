
from src.main.text_preprocessing_utilities import  *
from num2words import  num2words
if __name__ == "__main__":
    print("Main")
    #print(get_str_stopwords())
    # nlp_model = spacy.load("en_core_web_sm")
    # text = "email and quote and c_date"
    # tokens = get_spacy_tokens_from_raw_text(text, nlp_model)
    # for token in tokens:
    #     print(token)
    # tokens = lemmatize_spacy_tokens(tokens)
    # print(tokens)

    year_as_words = num2words(20131, to = 'year')
    print(year_as_words)




