import os
import pandas as pd

# IO functions
def read_txt_file(file_path):
    '''
    Return the content from the file from the given path. We assume the first line is the document title and the
    second line is document content

    :param file_path: path to the target file
    :return: a dictionary with 2 entries: title and content of the file
    :rtype: built-in python dictionary
    '''
    result = dict()
    with open(file_path, 'r', encoding='utf-8') as file_obj:
        result['title'] = file_obj.readline()
        result['content'] = file_obj.read()
    return result

def read_raw_data(main_directory_path):
    " read all files from all directories from the given path;  return a pandas df with 3 columns: document title, content and type (label) "
    df = pd.DataFrame(columns=['title','content','type'])
    directories = os.listdir(main_directory_path)

    new_files_contents = []

    for directory in directories:
        directory_path = main_directory_path + "\\" + directory
        files = os.listdir(directory_path)
        for file in files:
            file_path = directory_path + "\\" + file
            file_content = read_txt_file(file_path)

            whole_file_content_as_dict = pd.DataFrame({'title':file_content['title'], 'content':file_content['content'], 'type':directory}, index = [0])
            new_files_contents.append(whole_file_content_as_dict)

    df = pd.concat([df] + new_files_contents, ignore_index=True)

    return df

# data_root_path = "data"
# df = read_raw_data(data_root_path)
# df