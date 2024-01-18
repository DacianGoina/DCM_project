from src.main.io_utilities import *

from simpletransformers.classification import ClassificationModel

def load_transformer():
    roberta_dcm = import_binary_object('../../model_objects/RoBERTa/roberta_dcm.pkl')

    return roberta_dcm

# IN: boolean reverse
# OUT: dict with labels to integers values
# if reverse is True, the data and keys are flipped
def get_type_substitutions(reverse = False):
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

    rev_type_substitution = {data_value:key_value for key_value, data_value in type_substitution.items()}

    if reverse is True:
        return rev_type_substitution

    return type_substitution


# predict but use content direct from a txt file
def predict_from_file(file_path):
    input_content = read_txt_file(file_path)

    return predict(input_content)


# perform prediction for a single text
# IN: str raw text
# OUT: predicted label
def predict(raw_text):
    raw_text = [raw_text]

    roberta_dcm = load_transformer()
    substitutions_dict = get_type_substitutions(reverse=True)
    predicted_label, raw_output = roberta_dcm.predict(raw_text)
    predicted_label = int(predicted_label[0])

    return substitutions_dict[predicted_label]


if __name__ == "__main__":
    print("RoBERTa Worker")
    #pred = predict_from_file("../../testing_files/history_doc.txt")
    pred = predict("Usain Bolt is a great runner")
    print(pred)