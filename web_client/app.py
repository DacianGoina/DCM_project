from flask import Flask, render_template, request, flash, redirect
import requests

app = Flask(__name__)



SECRET_KEY = '192b9bdd22ab9ed4d12e236c78afcb9a393ec15f71bbf5dc987d54727823bcbf'
app.secret_key = SECRET_KEY  # for session management, required for flash messages

def GET_request(endpoint_uri = None, request_headers = None, request_params = None): 
    response = requests.get(endpoint_uri, headers = request_headers, params = request_params, timeout=10)
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
        get_result = GET_request("https://catfact.ninja/fact")
        print(file_content)
        flash(get_result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
