import os
import pandas as pd
import json
import pickle
# IO functions

# IN: str file path
# OUT: dict with file content
def read_txt_file(file_path):
    '''
    Return the content from the file from the given path.

    :param file_path: path to the target file
    :return: a dictionary with 2 entries: title and content of the file
    :rtype: built-in python dictionary
    '''
    result = dict()
    with open(file_path, mode = 'r', encoding='utf-8') as file_obj:
        result['content'] = file_obj.read()
    return result

# IN: directory str path
# OUT: pandas dataframe with 3 cols: title, content, type
# the given directory have the structure:
#   main_directory
#       subdirectory_category1
#           file
#           file
#           ...
#       subdirectory_category2
#       subdirectory_category3
#       ...
def read_raw_data(main_directory_path):
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

            whole_file_content_as_dict = pd.DataFrame({'file_path':file_path, 'content':file_content['content'], 'type':directory}, index = [0])
            new_files_contents.append(whole_file_content_as_dict)

    df = pd.concat([df] + new_files_contents, ignore_index=True)

    return df


# IN: dict, key: str value, key: int value
# OUT: None
# save the given dictionary at the given path
def save_dict_to_json_file(data, output_file_path):
    if output_file_path.endswith(".json") is False:
        output_file_path = output_file_path + ".json"

    with open(output_file_path, mode =  'w', encoding='utf-8') as f:
        json.dump(data, f, indent = 2)

# IN: object to serialize, output file path
# OUT: nothing
# serialize the given object (e.g list, scaler, classifier) and save it to a binary file
def export_as_binary_obj(obj, output_file_path):
    if output_file_path.endswith(".pkl") is False:
        output_file_path = output_file_path + ".pkl"

    with open(output_file_path, mode = 'wb') as file:
        pickle.dump(obj, file)

# IN: input file path (assume it is for an pickle binary file)
# OUT: unserialized python object, or None in case that problems occur during file reading process
def import_binary_object(input_file_path):
    try:
        with open(input_file_path, mode = 'rb') as file:
            res_obj = pickle.load(file)
            return res_obj
    except BaseException as e:
        print(e)
        return None

