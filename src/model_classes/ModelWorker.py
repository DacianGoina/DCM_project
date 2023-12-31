'''
This work as a worker / servant. It receive a raw text, preprocess it, convert to a specific numerical feature format
and use the predefined classifiers to classify it using a voting system.

'''
from src.main.model_utilities import *
from src.main.preprocessing_flow import *
from src.main.text_preprocessing_utilities import *
from src.main.io_utilities import *
from src.model_classes.ModelManager import reverse_classifier_to_extractor_str
from src.model_classes.ModelManager import get_classifier_to_extractor_str

import os

# in a way, these should be env variables
EXTRACTORS_OBJECTS_DIRECTORY_PATH = "../../model_objects/features_extractors"
CLASSIFIERS_OBJECTS_DIRECTORY_PATH = "../../model_objects/classifiers"

# root function for Model Worker
# IN: raw_text for classification
# OUT: predicted label, inside a dict with additional information
def worker_execute(raw_text):
    # preprocessing: convert raw text to text tokens
    tokens_as_single_str = preprocess_input(raw_text)

    # prediction using voting system
    predictions_result = perform_prediction(tokens_as_single_str)

    return predictions_result

# IN: preprocessed text (as whole string)
# OUT: dict with the predicted label, top predicted labels for the given text, scores provided by all classifiers
# load extractor and classifiers, perform prediction and compute the highest occurred predicted label
def perform_prediction(processed_text):
    classifiers, extractors = import_model_objects()

    numerical_data = dict()
    predictions = dict(); # d[classifier_extractor] = predicted_label

    # convert preprocessed text into numerical features using the extractors
    for extractor_name in extractors.keys():
        extractor_obj = extractors[extractor_name]
        processed_text_copy = processed_text[:] # copy original element
        processed_text_copy = [processed_text_copy] # extractor expect a list
        transformed_data = extractor_obj.transform_data(processed_text_copy)
        numerical_data[extractor_obj.short_str()] = transformed_data

    # perform predictions
    for extractor_name, classifiers_as_list in classifiers.items():
        for concrete_classifier in classifiers_as_list:
            #predicted_label = concrete_classifier.predict(numerical_data[extractor_name])
            predicted_label = concrete_classifier.predict_probabilities(numerical_data[extractor_name])
            classifier_extractor_pair_name = get_classifier_to_extractor_str(concrete_classifier.short_str(), extractor_name)
            predictions[classifier_extractor_pair_name] = predicted_label

    prediction_results = dict()

    predicted_label_final, top_predicted_labels = voting_system(predictions, n_highest_probs=1)
    prediction_results['predicted_label'] = predicted_label_final
    prediction_results['top_predicted_labels'] = top_predicted_labels
    prediction_results['all_predictions'] = predictions

    return prediction_results

# IN: dict with prediction results; d[classifier_extractor_name] = accuracy score scores for all classes
# OUT: the predicted label
# predict the label based on a voting system considering labels with predicted probabilities from top n_highest_probs
def voting_system(dict_with_predictions, n_highest_probs = 2):
    # for every classification results, sort labels by prediction probabilities
    for classifier_name in dict_with_predictions.keys():
        resulted_probabilities = dict_with_predictions[classifier_name]
        resulted_probabilities = sorted(resulted_probabilities, key = lambda prediction : prediction[1], reverse=True)
        dict_with_predictions[classifier_name] = resulted_probabilities

    # count frequency of occurrence considering labels from first n_highest_probs
    highest_occurred_labels = dict(); # dict[label] = number of occurrences
    for classifier_name, resulted_probabilities in dict_with_predictions.items():
        for index in range(0, n_highest_probs):
            label_name = resulted_probabilities[index][0]
            if label_name in highest_occurred_labels.keys():
                highest_occurred_labels[label_name] = highest_occurred_labels[label_name] + 1
            else:
                highest_occurred_labels[label_name] = 1

    highest_occurred_labels = sorted(highest_occurred_labels.items(), reverse=True, key = lambda pair: pair[1])
    highest_occurred_labels = dict(highest_occurred_labels) # from list of tuples, back to dict
    # get label with maximum number of occurrences
    label_with_highest_occurrences = max(highest_occurred_labels.items(), key = lambda pair: pair[1])
    return label_with_highest_occurrences[0], highest_occurred_labels

# OUT: tuple with classifiers and features extractors
# dict with features extractor: d[extractor_name] = extractor object,
# dict with classifiers: d[extractor_name] = [classifiers trained on data provided by given extractor]

# import model objects: classifiers and feature extractors
def import_model_objects():
    # import features extractors
    features_extractors = import_features_extractors()

    # import classifiers
    classifiers = import_classifiers()

    return classifiers, features_extractors

# OUT: dict with features extractor: d[extractor_name] = extractor object,
def import_features_extractors():
    extractors_objects_paths = os.listdir(EXTRACTORS_OBJECTS_DIRECTORY_PATH)
    features_extractors = dict(); # key: extractor name, key data: extractor object itself

    for extractor_obj_file_name in extractors_objects_paths:
        extractor_full_path = os.path.join(EXTRACTORS_OBJECTS_DIRECTORY_PATH, extractor_obj_file_name)
        extractor = import_binary_object(extractor_full_path)
        features_extractors[extractor.short_str()] = extractor

    return features_extractors

# OUT: dict with classifiers: d[extractor_name] = [classifiers trained on data provided by given extractor]
def import_classifiers():
    classifiers_objects_paths = os.listdir(CLASSIFIERS_OBJECTS_DIRECTORY_PATH)
    classifiers = dict() # key: extractor name "attached" to the classifier; key data: list of classifiers that use given extractor
    for classifier_obj_file_name in classifiers_objects_paths:
        classifier_full_path = os.path.join(CLASSIFIERS_OBJECTS_DIRECTORY_PATH, classifier_obj_file_name)
        classifier = import_binary_object(classifier_full_path)
        classifier_name, extractor_name = reverse_classifier_to_extractor_str(classifier_obj_file_name[0:-4])

        if extractor_name in classifiers.keys():
            classifiers[extractor_name].append(classifier)
        else:
            classifiers[extractor_name] = [classifier]

    return classifiers


# IN: str
# OUT: str (list of str tokens joined with ' ')
# preprocess the raw text received as input
def preprocess_input(raw_text):
    nlp_model = get_nlp_model()
    tokens = apply_custom_tokenizer_iteratively(raw_text, nlp_model, iterations=4)
    tokens_as_single_str = str_tokens_to_str(tokens)

    return tokens_as_single_str

#OUT: dict, d[classifier_name] = dict with metrics and associated values
# read classifiers, compute model performance metrics using confusions matrices
def evaluate_classifiers():
    classifiers_dict = import_classifiers()
    classifiers = []
    for method_name, method_classifiers in classifiers_dict.items():
        for classifier in method_classifiers:
            classifier_full_name = get_classifier_to_extractor_str(classifier.short_str(), method_name)
            classifiers.append((classifier_full_name, classifier))

    results = dict()
    for classifier_full_name, classifier in classifiers:
        results[classifier_full_name] = get_model_evaluation_metrics(classifier.get_confusion_matrix())

    return results


# IN: file path to a txt file (raw text)
# OUT: predicted label, inside a dict with additional information
# read text from the file given by the path, perform prediction on it and return the resulted dictionary
def predict_input_from_file(file_path):
    text_input = read_txt_file(file_path)
    prediction_result = worker_execute(text_input)
    return prediction_result

if __name__ == "__main__":
    print("Model Worker")

    sport = "No excuses. It is a deeply ingrained part of Ange Postecoglou’s management style. Just keep fighting. And remember to be grateful. Anyone who plays football professionally is living the dream. And yet there have to be times when the Tottenham manager wants to reach for something, a little context. Now, as he navigates his first Christmas and new year programme in the Premier League, with his team feeling the burn, is one of those times. Tottenham goalkeeper Hugo Lloris Spurs looked shattered for most of the first 80 minutes at Brighton on Thursday, second best in all areas, 4-0 down, staring at humiliation. Which is what made the late rally to 4-2 so remarkable, why Postecoglou was keen to praise his players to the hilt. Spurs went close to a third goal; they hinted at the wildest of comebacks."

    business_doc_path = "../../testing_files//business_doc.txt"
    history_text_path = "../../testing_files//history_doc.txt"
    file_path_basketball = "../../testing_files//basketball.txt"
    # print(predict_input_from_file(file_path_basketball))

    result1 = worker_execute(sport)
    print(result1)

    print('\n\n')

    print(evaluate_classifiers())
