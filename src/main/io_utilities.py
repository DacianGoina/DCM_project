import os
import pandas as pd

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
    with open(file_path, 'r', encoding='utf-8') as file_obj:
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

# data_root_path = "data"
# df = read_raw_data(data_root_path)
# df