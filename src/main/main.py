from src.main.text_preprocessing_utilities import  *
from num2words import  num2words
from src.main.preprocessing_flow import *
from src.main.io_utilities import *
from sklearn.feature_extraction.text import CountVectorizer
from pickle import dump
from pickle import load
if __name__ == "__main__":
    print("Main")

    TOKEN_PATTERN_v = "\S+"
    cv = CountVectorizer(lowercase = False, stop_words = None, token_pattern = TOKEN_PATTERN_v, preprocessor = None, tokenizer = None)

    in_list = ['house for my family',
               'cat like house house',
               'house for cat'
    ]
    #cv.fit(in_list)
    # cv.transform(in_list)
    # print(cv.vocabulary_)
    # print(cv.transform(in_list).toarray())
    # print("-----------------")
    # print(cv.transform(['house caca maca']).toarray())

    # res = cv.fit_transform(in_list)
    # print(cv.vocabulary_)
    # print(sorted(cv.vocabulary_.copy().items(), key = lambda x: x[1]))
    # print(res.toarray())
    # print('-'*20)
    # res2 = cv.transform(['mickey house cat agagagag'])
    # print(res2.toarray())
    #
    # dump(cv, open('count_vec.pkl', 'wb'))

    # cv2 = import_binary_object("count_vec.pkl")
    # print(cv2)
    # print(cv2.vocabulary_)
    # print(cv2.transform(["house my cat cool"]).toarray())

    # model1 = get_nlp_model()
    # text = "\"the https://www.kaggle.com is a good website for datasets!"
    # tokens = get_spacy_tokens_from_raw_text(text, model1)
    # tokens = spacy_tokens_to_str_tokens(tokens)
    # print(tokens)
    # print(validators.url("https://www.kaggle.com"))
    # print(validators.url("https://www.kaggle.com/"))
    # print(validators.url("http://www.kaggle.com"))
    # print(bool(validators.url("www.kaggle.com")))
    # print('*'*20)
    # print(urlparse("https://www.kaggle.com"))
    # print(urlparse("https://www.kaggle.com/"))
    # print(urlparse("http://www.kaggle.com"))
    # print(urlparse("www.kaggle.com"))
    # print('-'*20)
    # print(is_valid_url("https://www.kaggle.com"))
    # print(is_valid_url("https://www.kaggle.com/"))
    # print(is_valid_url("http://www.kaggle.com"))
    # print(is_valid_url("www.kaggle.com"))
    # print(is_valid_url("https://example.com/resource/path?query=param"))

    # tokens = str_urls_to_url_tag([initial_case_letter,'flies','writen','researcher','www.kaggle.com', 'ames.arc.nasa.gov', 'https://www.kaggle.com/', 'https://example.com/resource/path?query=param'])
    # tokens = str_tokens_to_spacy_tokens(tokens,get_nlp_model())
    # #print(lemmatize_spacy_tokens(tokens))
    # for token in tokens:
    #     print(token.lemma_)
    # print(urlparse("ames.arc.nasa.gov"))
    # print(urlparse("not an url"))



    # regex = re.compile(
    #     r'^(?:http|ftp)s?://' # http:// or https://
    #     r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
    #     r'localhost|' #localhost...
    #     r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
    #     r'(?::\d+)?' # optional port
    #     r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    #
    # print(re.match(regex, "http://www.example.com") is not None) # True
    # print(re.match(regex, "example.com") is not None)
    # print(re.match(regex, "ames.arc.nasa.gov") is not None)

    # print(is_url("http://www.example.com"))
    # print(is_url("example.com"))

    #read_preprocess_and_export("../data","file_name_v6.csv", preprocessing_iterations=4)