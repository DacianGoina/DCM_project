
from text_preprocessing_utilities  import *
from preprocessing_flow import  *
from num2words import num2words
if __name__ == "__main__":
    print("Main")

    read_preprocess_and_export(directory_path='../data', output_file_name='file_name_v4.csv')
