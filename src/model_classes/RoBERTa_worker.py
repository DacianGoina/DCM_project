from src.main.io_utilities import *

from simpletransformers.classification import ClassificationModel

def load_transformer():
    roberta_dcm = import_binary_object('../../model_objects/RoBERTa/roberta_dcm.pkl')

    return roberta_dcm


def get_type_substitutions():
    type_substitution = {
        'business':0,
        'entertainment':1,
        'food':2,
        'graphics':3,
        'historical':4,
        'medical':5,
        'politics':6,
        'space':7,
        'sport':8,
        'technology':9
    }

    return type_substitution

def predict(file_path):

    roberta_dcm = load_transformer()

    input_content = read_txt_file(file_path)
    input_content = [input_content]
    predicted_label, raw_output = roberta_dcm.predict(input_content)

    print(type(roberta_dcm))
    print(predicted_label)

    return predicted_label

if __name__ == "__main__":
    print("RoBERTa Worker")
    predict("../../testing_files/history_doc.txt")