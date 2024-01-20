from flask import Flask, render_template, request, flash, redirect
from src.model_classes.ModelWorker import perform_preprocessing_and_predict, import_model_objects
import requests


app = Flask(__name__)

SECRET_KEY = '192b9bdd22ab9ed4d12e236c78afcb9a393ec15f71bbf5dc987d54727823bcbf'
app.secret_key = SECRET_KEY  # for session management, required for flash messages

MODEL_CLASSIFIERS = {}
MODEL_EXTRACTORS = {}

def GET_request(endpoint_uri = None, request_headers = None, request_params = None):
    '''
    GET request general function
    '''
    response = requests.get(endpoint_uri, headers = request_headers, params = request_params, timeout=10)
    if response.status_code != 200:
        return("GET failed")
    else:
        json_res = response.json()
        return json_res

def prepare_string_result(dict_result):
    '''
    Prepare string to be displayed on the screen as a result. Start from the dictionary with prediction results;
    :param dict_result:
    :return:
    '''
    res = "Document type: " + dict_result['predicted_label']
    res = res + "\n\nLabel voting situation:\n"

    for label, counter in dict_result['top_predicted_labels'].items():
        res = res + label + ": " + str(counter) + "\n"

    return res

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    '''
    Handle a file upload action: receive the file, read his content, convert to string and perform the prediction over it.
    :return:
    '''
    if request.method == 'POST':
        print(request.files)
        # check if 'file' key is in the request
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        
        
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            flash('No selected file')
            return render_template('index.html')
            
        # access the file
        file_name = file.filename
        file_content = file.read()
        file_content = file_content.decode('utf-8', errors='ignore')
        dict_result = perform_preprocessing_and_predict(file_content.strip(), False, MODEL_CLASSIFIERS, MODEL_EXTRACTORS)
        result_as_string = prepare_string_result(dict_result)
        flash(result_as_string)
    return render_template('index.html')

if __name__ == '__main__':
    MODEL_CLASSIFIERS, MODEL_EXTRACTORS = import_model_objects()
    print("App started")
    app.run(debug=False)
