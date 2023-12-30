import os
import pandas as pd
import json
import pickle
# IO functions

# IN: str file path
# OUT: string with file content
def read_txt_file(file_path):
    '''
    Return the content from the file from the given path.

    :param file_path: path to the target file
    :return: file content
    :rtype: built-in python string
    '''
    result = None
    with open(file_path, mode = 'r', encoding='utf-8') as file_obj:
        result = file_obj.read()
    return result


def read_raw_data(main_directory_path):
    '''
    Creates a dataframe with the structure of a directory
    The given directory has a structure like:
    main_directory
        subdirectory_category1
            file
            file
            ...
        subdirectory_category2
        subdirectory_category3
        ...
    :param main_directory_path: the path given of the main directory
    :return: a dataframe with 3 columns: title, content, type
    :rtype: pandas.core.frame.DataFrame
    '''
    " read all files from all directories from the given path;"
    # return a pandas df with 3 columns: document title, content and type (label) "
    df = pd.DataFrame(columns=['file_path','content','type'])
    directories = os.listdir(main_directory_path)

    new_files_contents = []

    for directory in directories:
        directory_path = main_directory_path + "\\" + directory
        files = os.listdir(directory_path)
        for file in files:
            file_path = directory_path + "\\" + file
            file_content = read_txt_file(file_path)

            whole_file_content_as_dict = pd.DataFrame({'file_path':file_path, 'content':file_content, 'type':directory}, index = [0])
            new_files_contents.append(whole_file_content_as_dict)

    df = pd.concat([df] + new_files_contents, ignore_index=True)

    return df


# IN: dict, key: str value, key: int value
# OUT: None
# save the given dictionary at the given path
def save_dict_to_json_file(data, output_file_path):
    '''
    Function to save a dictionary in a json file at a given path
    :param data: dictionary that contains pairs of (key, str_value/s), (key, int_value/s)
    :param output_file_path: path where the json will be saved
    :return: None
    '''
    if output_file_path.endswith(".json") is False:
        output_file_path = output_file_path + ".json"

    with open(output_file_path, mode =  'w', encoding='utf-8') as f:
        json.dump(data, f, indent = 2)


def export_as_binary_obj(obj, output_file_path):
    '''
    Function to serialize a given object and save it in a binary file
    :param obj: an object, can be a list, scaler, classifier
    :param output_file_path: path where the json will be saved
    :return: None
    '''
    if output_file_path.endswith(".pkl") is False:
        output_file_path = output_file_path + ".pkl"

    with open(output_file_path, mode = 'wb') as file:
        pickle.dump(obj, file)


def import_binary_object(input_file_path):
    '''
    Function used to import a pickle binary file
    :param input_file_path: the path of the file that need to be imported
    :return: unserialized python object, or None in case of errors during file reading
    '''
    try:
        with open(input_file_path, mode = 'rb') as file:
            res_obj = pickle.load(file)
            return res_obj
    except BaseException as e:
        print(e)
        return None

