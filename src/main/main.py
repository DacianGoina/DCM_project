
from src.main.text_preprocessing_utilities import  *
from num2words import  num2words
from src.main.preprocessing_flow import *
if __name__ == "__main__":
    print("Main")
    print(num2words(6.62618e-34))
    #read_preprocess_and_export("../data","file_name_v5.csv", preprocessing_iterations=4)