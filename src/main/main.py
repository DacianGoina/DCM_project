from src.main.text_preprocessing_utilities import  *
from num2words import  num2words
from src.main.preprocessing_flow import *
from src.main.io_utilities import *
from sklearn.feature_extraction.text import CountVectorizer
from pickle import dump
from pickle import load
if __name__ == "__main__":
    print("Main")
    #read_preprocess_and_export("../data","file_name_v6.csv", preprocessing_iterations=4)