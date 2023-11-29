from io_utilities import *
from num2words import num2words
from spacy.lang.en import English
if __name__ == "__main__":
    # data_root_path = "../data"
    # df = read_raw_data(data_root_path)
    # print(df)
    parser = English()
    sentence = 'Winn-Dixie, once among the most profitable of US grocers, said Chapter 11 protection would enable it to successfully restructure. It said its 920 stores would remain open, but analysts said it would most likely off-load a number of sites. The Jacksonville, Florida-based firm has total debts of $1.87bn (£980m). In its bankruptcy petition it listed its biggest creditor as US foods giant Kraft Foods, which it owes $15.1m.'
    mytokens = parser(sentence)

    for token in mytokens:
        print(token.text, token.pos_)

    #print(num2words("25", to="currency"))

