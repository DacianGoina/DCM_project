from flask import Flask, render_template, request, flash, redirect
import requests

app = Flask(__name__)

app.secret_key = 'super_secret_key'  # for session management, required for flash messages

def dummy_GET_REQUEST():
    endpoint_url = "https://catfact.ninja/fact"
    response = requests.get(endpoint_url, headers = None, params = None, timeout=10)
    if response.status_code != 200:  
        return("GET failed")
    else:
        json_res = response.json()
        print(json_res['fact'])
        return json_res['fact']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
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
            
        # acces the file
        file_name = file.filename
        file_content = file.read()
        
        # do something with file's text
        get_result = dummy_GET_REQUEST()
        print(file_content)
        flash(get_result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
