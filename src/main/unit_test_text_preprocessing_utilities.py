import unittest
import spacy
from src.main.text_preprocessing_utilities import *

class UnitTests(unittest.TestCase):

    def test_is_str_numeric(self):
        '''
        Unit test for is_str_numeric when it has numerical/non numerical values
        '''
        self.assertTrue(is_str_numeric("123"))
        self.assertTrue(is_str_numeric("3.14"))
        self.assertTrue(is_str_numeric("-42"))

        self.assertFalse(is_str_numeric("abc"))
        self.assertFalse(is_str_numeric("1a2b3c"))
        self.assertFalse(is_str_numeric("3.14.15"))
        self.assertFalse(is_str_numeric("infinity"))

    def test_is_str_valid_date(self):
        '''
        Unit test for valid/non valid dates
        '''
        self.assertTrue(is_str_valid_date("2022-01-01"))
        self.assertTrue(is_str_valid_date("2023-12-31"))
        self.assertTrue(is_str_valid_date("1990-05-20"))

        self.assertFalse(is_str_valid_date("2022-13-01")) # invalid month
        self.assertFalse(is_str_valid_date("2023-02-30"))  # invalid day for February
        self.assertFalse(is_str_valid_date("2021-12-32"))  # invalid day for December
        self.assertFalse(is_str_valid_date("2022-02-29"))  # non-leap year

    def test_is_str_fraction(self):
        '''
        Unit test for valid types of fractions and invalid inputs
        '''
        self.assertTrue(is_str_fraction("3/4"))
        self.assertTrue(is_str_fraction("2/5"))
        self.assertTrue(is_str_fraction("7/10"))

        self.assertFalse(is_str_fraction("1/ 3"))  # contains one space
        self.assertFalse(is_str_fraction("2 / 3"))  # contains spaces
        self.assertFalse(is_str_fraction("abc"))  # value that is not a fraction

    def test_get_lowercase_words_from_str(self):
        '''
        Unit test for different cases that includes multiple lowercase words and mixed words (formed with lower and upper cases)
        '''
        self.assertEqual(get_lowercase_words_from_str("hello world"), ["hello", "world"])
        self.assertEqual(get_lowercase_words_from_str("   python "), ["python"])
        self.assertEqual(get_lowercase_words_from_str("one two tree"), ["one", "two", "tree"])
        self.assertEqual(get_lowercase_words_from_str("Hello World"),["ello", "orld"])
        self.assertEqual(get_lowercase_words_from_str("PyThoN ProgRaMMing"), ['y', 'ho', 'rog', 'a', 'ing'])
        self.assertEqual(get_lowercase_words_from_str("ONE TwO trEE"), ['w', 'tr'])

    def test_to_lowercase(self):
        '''
        Unit test for text lowercase
        '''
        self.assertEqual(to_lowercase("ThIs Is A MiXeD CaSe TeXT"), "this is a mixed case text")
        self.assertEqual(to_lowercase(""), "")

    def test_remove_excessive_space(self):
        '''
        Unit test to see if the spaces are eliminated in strings with multiple spaces
        '''
        self.assertEqual(remove_excessive_space("    hello world              "), "hello world")
        self.assertEqual(remove_excessive_space("                  python programming     "), "python programming")
        self.assertEqual(remove_excessive_space("\tkeep coding\t"), "keep coding")

        self.assertEqual(remove_excessive_space("hello"), "hello")
        self.assertEqual(remove_excessive_space("hello world"), "hello world")
        self.assertEqual(remove_excessive_space("one two tree"), "one two tree")

    def test_remove_spacy_punctuations(self):
        '''
        Unit test for list with multiple punctuations and null list
        '''
        input = "Hello , world ! What is ? yes ."
        nlp = spacy.load("en_core_web_md")
        result = remove_spacy_punctuations(nlp(input))
        expected_result = list(nlp("Hello world What is yes"))

        self.assertEqual(result, expected_result)
        self.assertEqual(remove_spacy_punctuations([]), [])

    def test_lemmatize_spacy_tokens(self):
        '''
        Unit test for list of words that contains lematized words
        '''
        nlp = spacy.load("en_core_web_sm")

        input_text = "This is a sample sentence with some lemmatizable words"
        input_tokens = nlp(input_text)

        self.assertEqual(lemmatize_spacy_tokens(input_tokens), ['this', 'be', 'a', 'sample', 'sentence', 'with', 'some', 'lemmatizable', 'word'])
        self.assertEqual(lemmatize_spacy_tokens([]), [])

    def test_str_tokens_remove_stopwords(self):
        '''
        Unit test for removing stopwords from a list of tokens
        '''
        result = str_tokens_remove_stopwords(["this", "is", "a", "test", "sentence"])
        expected_result = ['test', 'sentence']
        self.assertEqual(result, expected_result)

        self.assertEqual([], [])

    def  test_get_str_tokens_freq(self):
        '''
        Unit test for function that gets the tokens frequencies
        '''
        input_tokens = ["apple", "banana", "apple", "orange", "banana", "apple"]
        self.assertEqual(get_str_tokens_freq(input_tokens), {"apple": 3, "banana": 2, "orange": 1})
        self.assertEqual(get_str_tokens_freq([]), {})

    def test_get_str_tokens_freq_for_lists(self):
        '''
        Unit test for function thst the tokens frequencies from list of lists
        '''
        input_lists = [["apple", "banana", "apple", "orange", "banana", "apple"],
                       ["banana", "orange", "orange", "kiwi", "kiwi"],
                       ["apple", "apple", "kiwi", "banana"]]
        self.assertEqual(get_str_tokens_freq_for_lists(input_lists), {"apple": 5, "banana": 4, "orange": 3, "kiwi": 3})
        self.assertEqual(get_str_tokens_freq_for_lists([]), {})

    def test_get_rare_tokens(self):
        '''
        Unit test for function that extracts rare tokens
        '''
        input_dict = {"apple": 5, "banana": 4, "orange": 3, "kiwi": 2, "pear": 1}
        threshold = 3
        self.assertEqual(get_rare_tokens(input_dict, threshold), {"kiwi": 2, "orange": 3, "pear": 1})
        self.assertEqual(get_rare_tokens({}, 1), {})

    def test_handle_rare_str_tokens(self):
        '''
        Unit test for function that eliminates or replace rare words
        '''
        input_tokens = ["apple", "banana", "orange", "kiwi", "pear"]
        input_dict_of_freq = {"apple": 5, "banana": 4, "orange": 3, "kiwi": 2, "pear": 1}
        replace_with = None
        self.assertEqual(handle_rare_str_tokens(tokens=input_tokens, dict_of_freq=input_dict_of_freq, replace_with=replace_with), [])

        input_tokens = ["apple", "banana", "orange", "kiwi", "pear"]
        input_dict_of_freq = {"apple": 5, "banana": 4}
        replace_with = "RARE"
        self.assertEqual(handle_rare_str_tokens(tokens=input_tokens, dict_of_freq=input_dict_of_freq, replace_with=replace_with), ['[UNK]', '[UNK]', 'orange', 'kiwi', 'pear'])

    def test_spacy_tokens_pos(self):
        '''
        Unit test for function that creates a list of tuples containing token and it's position
        '''
        nlp = spacy.load("en_core_web_sm")
        sentence = "The quick brown fox jumps over the lazy dog."
        spacy_tokens = list(nlp(sentence))
        result = spacy_tokens_pos(spacy_tokens)
        expected_result = [(token, token.pos_) for token in spacy_tokens]
        self.assertEqual(result, expected_result)

        self.assertEqual(spacy_tokens_pos([]), [])

    def test_get_spacy_tokens_from_raw_text(self):
        '''
        Unit test for functions that extracts spacy tokens from raw text
        '''
        nlp = spacy.load("en_core_web_sm")
        raw_text = "This is a sample sentence."
        result = get_spacy_tokens_from_raw_text(raw_text, nlp)
        expected_result = list(nlp(raw_text))
        print(type(result), type(expected_result))
        self.assertEqual(result, expected_result)

        self.assertEqual(get_spacy_tokens_from_raw_text("", spacy.blank("en")), [])

    def test_str_remove_junk_spaces(self):
        '''
        Unit test for functions that removes junk spaces in list
        '''
        tokens_with_junk_spaces = ['  word1  ', 'word2\t', '\rword3\n', 'word4\n\n  ']
        result = str_remove_junk_spaces(tokens_with_junk_spaces)
        expected_result = ['word1', 'word2', 'word3', 'word4']
        self.assertEqual(result, expected_result)

        self.assertEqual(str_remove_junk_spaces([]), [])

    def test_str_years_to_spoken_words(self):
        '''
        Unit test for function that transform numerical year in text year
        '''
        tokens_with_years = ['The', 'year', '1990', 'marked', 'the', 'beginning', 'of', 'a', 'new', 'era']
        result = str_years_to_spoken_words(tokens_with_years)
        expected_result = ['The', 'year', 'nineteen', 'ninety', 'marked', 'the', 'beginning', 'of', 'a', 'new', 'era']
        self.assertEqual(result, expected_result)

        tokens_with_years = ['1992', '463', '1234', '2', '13', '45', '1432']
        result = str_years_to_spoken_words(tokens_with_years)
        expected_result =['nineteen', 'ninety', 'two', '463', 'twelve', 'thirty', 'four', '2', '13', '45', 'fourteen', 'thirty', 'two']
        self.assertEqual(result, expected_result)

    def test_str_numeric_values_to_spoken_words(self):
        '''
        Unit test for function that transform numerical values in test
        '''
        input_tokens = ['54', '2.5', '10', '3.14', 'abc', '123.45']
        result = str_numeric_values_to_spoken_words(input_tokens)
        expected_output =['fifty', 'four', 'two', 'point', 'five', 'ten', 'three', 'point', 'one', 'four', 'abc', 'one', 'hundred', 'and', 'twenty', 'three', 'point', 'four', 'five']
        self.assertEqual(result, expected_output)

        self.assertEqual(str_numeric_values_to_spoken_words([]), [])

    def test_str_ordinal_numbers_to_spoken_words(self):
        '''
        Unit test for functions that transform ordinal number in text
        '''
        input_tokens = ['1', 'st', '2', 'nd', '3', 'rd', '4', 'th', 'abc', '5', 'th']
        result = str_ordinal_numbers_to_spoken_words(input_tokens)
        expected_output = ['first', 'second', 'third', 'fourth', 'abc', 'fifth']
        self.assertEqual(result, expected_output)

        self.assertEqual(str_ordinal_numbers_to_spoken_words([]), [])

    def test_str_currency_to_spoken_words(self):
        '''
        Unit test for functions that transform currency in text
        '''
        input_tokens = ['$', 'EUR', '£', 'CHF', '5', 'USD', '10%', 'JPY', '15', 'AUD']
        result = str_currency_to_spoken_words(input_tokens)
        expected_output = ['dollars', 'euros', 'sterling', 'swiss francs', '5', 'dollars', '10%', 'yens', '15', 'dollars']
        self.assertEqual(result, expected_output)

        self.assertEqual(str_currency_to_spoken_words([]), [])

    def test_str_remove_common_chars(self):
        '''
        Unit test for functions that removes common chars such as \
        '''
        input_tokens = ['apple', '"banana"', '\'orange\'', 'cherry', 'grape', '\'mango\'', 'kiwi']
        result = str_remove_common_chars(input_tokens)
        expected_output = ['apple', '"banana"', "'orange'", 'cherry', 'grape', "'mango'", 'kiwi']
        self.assertEqual(result, expected_output)

        self.assertEqual(str_remove_common_chars([]), [])


    def test_remove_str_tokens_len_less_than_threshold(self):
        '''
        Unit test for functions that removes tokens with length smaller than a threshold
        '''
        input_tokens = ['apple', 'banana', 'orange', 'cherry', 'grape', 'mango', 'kiwi']
        threshold_value = 5
        result = remove_str_tokens_len_less_than_threshold(input_tokens, threshold_value)
        expected_output = ['apple', 'banana', 'orange', 'cherry', 'grape', 'mango']
        self.assertEqual(result, expected_output)

        threshold_value = 6
        expected_output = ['apple', 'banana', 'orange', 'cherry', 'grape', 'mango']
        self.assertEqual(result, expected_output)

    def test_str_fraction_to_spoken_words(self):
        '''
        Unit test for function that transform fractions to text
        '''
        input_tokens = ['1/2', '2/3', '3/4', '5/8', '1/4']
        result = str_fraction_to_spoken_words(input_tokens)
        expected_output = ['one', 'half', 'two', 'third', 'three', 'fourth', 'five', 'eighth', 'one', 'fourth']
        self.assertEqual(result, expected_output)

        input_tokens = ['2/5', '3/7', '4/9', '6/11', '1/3']
        result = str_fraction_to_spoken_words(input_tokens)
        expected_output = ['two', 'fifth', 'three', 'seventh', 'four', 'ninth', 'six', 'eleventh', 'one', 'third']
        self.assertEqual(result, expected_output)

    def test_str_emails_to_email_tag(self):
        '''
        Unit test for function that replace email with an email tag
        '''
        input_tokens = ['john.doe@example.com', 'alice.smith@gmail.com', 'info@company.com']
        result = str_emails_to_email_tag(input_tokens)
        expected_output = ['email', 'email', 'email']
        self.assertEqual(result, expected_output)

        input_tokens = ['customer', 'support@helpdesk.com', 'team', 'sales@business.co', 'contact']
        result = str_emails_to_email_tag(input_tokens)
        expected_output = ['customer', 'email', 'team', 'email', 'contact']
        self.assertEqual(result, expected_output)

    def test_str_dates_to_date_tag(self):
        '''
        Unit test for function that replace dates to tag date
        '''
        input_tokens = ['2022-01-01', 'apple', '2023-12-31', 'banana']
        result = str_dates_to_date_tag(input_tokens)
        expected_output = [calendar_date_tag, 'apple', calendar_date_tag, 'banana']
        self.assertEqual(result, expected_output)

        input_tokens = ['apple', 'orange', 'banana']
        result = str_dates_to_date_tag(input_tokens)
        expected_output = ['apple', 'orange', 'banana']
        self.assertEqual(result, expected_output)

    def test_str_tokens_to_spacy_tokens(self):
        '''
        Unit test for function that transforms str tokens in spacy tokens
        '''
        nlp = spacy.load("en_core_web_sm")
        input_tokens = ['apple', 'orange', 'banana']
        result = str_tokens_to_spacy_tokens(input_tokens, nlp)
        expected_output = list(nlp(input_token) for input_token in input_tokens )
        self.assertEqual(result, expected_output)

        self.assertEqual(str_tokens_to_spacy_tokens([], self.nlp_model), [])

    def test_spacy_tokens_to_str_tokens(self):
        '''
        Unit test for function that transforms spacy tokens in str tokens
        '''
        nlp = spacy.load("en_core_web_sm")
        input_tokens = nlp("apple orange banana")
        result = spacy_tokens_to_str_tokens(input_tokens)
        expected_output = ["apple", "orange", "banana"]
        self.assertEqual(result, expected_output)

        self.assertEqual(spacy_tokens_to_str_tokens([]), [])

    def test_str_tokens_to_str(self):
        '''
        Unit test for function that transforms str tokens in str
        '''
        input_tokens = ["apple", "orange", "banana"]
        result = str_tokens_to_str(input_tokens)
        expected_result = "apple orange banana"
        self.assertEqual(result, expected_result)

        self.assertEqual(str_tokens_to_str([]), '')

    def test_str_tokens_replace_quote_with_tag(self):
        '''
        Unit tests for function that replace a given symbol with a specified tag
        '''
        result = str_tokens_replace_symbol_with_tag(["\"", "This", "is", "\"", "a", "statement", "\"", "!"], "\"", "[tag]")
        expected_result = ["[tag]", "This", "is", "[tag]", "a", "statement", "[tag]", "!"]
        self.assertEqual(result, expected_result)

        self.assertEqual(str_tokens_replace_symbol_with_tag([], "", ""), [])

    def test_str_tokens_numbers_with_separators_to_spoken_words(self):
        '''
        Unit test for function that transform numbers that contains separators in spoken words
        '''
        input = ['there', 'are', '10,500,400', 'variants', 'and', '78,392', 'methods']
        expected_output = ['there', 'are', 'ten', 'million', 'five', 'hundred', 'thousand', 'four', 'hundred', 'variants', 'and', 'seventy', 'eight', 'thousand', 'three', 'hundred', 'and', 'ninety', 'two', 'methods']
        self.assertEqual(str_tokens_numbers_with_separators_to_spoken_words(input), expected_output)

        self.assertEqual(str_tokens_numbers_with_separators_to_spoken_words(['10,3,alb']), ['10,3,alb'])

    def test_str_number_with_separators_to_integer_number(self):
        '''
        Unit test for function that transform string numerical values separated by commas in integer values
        '''
        input = "10,500,205"
        expected_output = 10500205
        self.assertEqual(str_number_with_separators_to_integer_number(input), expected_output)

        input = "10500205"
        expected_output = 10500205
        self.assertEqual(str_number_with_separators_to_integer_number(input), expected_output)

    def test_split_and_gather_str_tokens_by_separator(self):
        '''
        Unit test for function that split all tokens by a given separator in multiple tokens and replace that tokens with the split values
        '''
        input = ['this', 'is', 'a', 'world,for', 'everyone,who,lives', 'here']
        expected_output = ['this', 'is', 'a', 'world', 'for', 'everyone' , 'who', 'lives', 'here']
        self.assertEqual(split_and_gather_str_tokens_by_separator(input, separator=","), expected_output)

        self.assertEqual(split_and_gather_str_tokens_by_separator(['nothing_to_split'], separator=","), ['nothing_to_split'])

    def test_is_6digits_date(self):
        '''
        Unit test to check if a string is a calendar date formed with  or not
        '''
        self.assertTrue(is_6digits_date("12-04-20"))
        self.assertTrue(is_6digits_date("27/07/09"))
        self.assertTrue(is_6digits_date("07.04.23"))


        self.assertFalse(is_6digits_date("12.09 20"))
        self.assertFalse(is_6digits_date("this"))
        self.assertFalse(is_6digits_date("14.05.1"))

    def test_str_6digits_dates_to_date_tag(self):
        '''
        Unit test for function that replace 6 digits dates appearances with c_date
        '''
        input = ["This", "is", "a", "sample", "date", "14.05.93", "in", "the", "text"]
        expected_output = ["This", "is", "a", "sample", "date", "c_date", "in", "the", "text"]
        self.assertEqual(str_6digits_dates_to_date_tag(input), expected_output)

        input =  ["No", "date", "here", "123456", "567890", "abc"]
        expected_output =  ["No", "date", "here", "123456", "567890", "abc"]
        self.assertEqual(str_6digits_dates_to_date_tag(input), expected_output)

    def test_is_valid_url(self):
        '''
        Unit test for function that checks if a string is a valid URL or not
        '''
        self.assertTrue(is_valid_url("http://www.example.com"))
        self.assertTrue(is_valid_url("https://example.org/page"))
        self.assertTrue(is_valid_url("http://www.google.com"))

        self.assertFalse(is_valid_url("http://"))
        self.assertFalse(is_valid_url("ftpl..example.com"))
        self.assertFalse(is_valid_url("not an url"))

    def test_is_valid_resource(self):
        '''
        Unit test for function that checks if an url is in small notation
        '''
        self.assertTrue(is_valid_resource("ames.arc.nasa.gov"))
        self.assertTrue(is_valid_resource("prf.ssl.plf"))

        self.assertFalse(is_valid_resource("1234.resource.com"))
        self.assertFalse(is_valid_resource("invalid-resource"))

    def test_str_urls_to_url_tag(self):
        '''
        Unit test for function that replace URL with tag
        '''
        input = ["Visit", "our", "website", "at", "https://www.example.com", "for", "more", "information"]
        expected_output = ["Visit", "our", "website", "at", "url", "for", "more", "information"]
        self.assertEqual(str_urls_to_url_tag(input), expected_output)

        input = ["Invalid", "URL", "format", "www.invalid-url.com"]
        expected_output = ["Invalid", "URL", "format", "www.invalid-url.com"]
        self.assertEqual(str_urls_to_url_tag(input), expected_output)

    def test_str_initial_case_to_tag(self):
        input = ["This", "is", "A.", "sample", "text", "including", "a.", "representation", "of", "e."]
        expected_output = ["This", "is", "init_case", "sample", "text", "including", "init_case", "representation", "of", "init_case"]
        self.assertEqual(str_initial_case_to_tag(input), expected_output)

        input = ["Invalid", "case", "format", "X", "."]
        expected_output = ["Invalid", "case", "format", "X", "."]
        self.assertEqual(str_initial_case_to_tag(input), expected_output)

if __name__ == '__main__':
    unittest.main()



