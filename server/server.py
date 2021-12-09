from flask import Flask, request, abort, jsonify, make_response
import time
import os
import numpy as np
import scipy.io
from feature_model_extraction import 
from werkzeug.utils import secure_filename

app = Flask(__name__)
# app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.mat']
app.config['UPLOAD_PATH'] = 'uploadedData'

@app.route('/')
def index():
    return "Biometric Liveliness Project - ASU MC CSE535 - Group 23"

@app.route('/classification', methods=['POST'])
def upload_files():
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            abort(400)
        
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
        
        mat = scipy.io.loadmat(os.path.join(app.config['UPLOAD_PATH'], filename))
        dataset = mat['data']
        print("dataset shape is:")
        print(dataset.shape)
        if dataset.shape[0] == 1:
            print("file format is correct")
        else:
            print("file format is incorrect")

        time.sleep(1)

        response_body = {
            "classification_result": 1
        }
        print("Final Output")
        print(jsonify(response_body))
        res = make_response(jsonify(response_body), 200)
        return res
    else:
        abort(400)

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5001,threaded = False)


# https://blog.miguelgrinberg.com/post/handling-file-uploads-with-flask