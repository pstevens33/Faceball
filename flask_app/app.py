from flask import Flask, render_template, request, send_from_directory, jsonify, Response, g, flash, redirect
from flask import make_response
from functools import wraps, update_wrapper
from datetime import datetime
import os
import json
from os import listdir
from os.path import isfile, join
from werkzeug import secure_filename
import pandas as pd
import time

import numpy as np
from keras.models import load_model

import sys
sys.path.insert(0, '../src')
from project_faces_local import project_face
from image_processing_web import process_image


app = Flask(__name__)
app.secret_key = "super secret key"



# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'uploads/'
# These are the extension that we are accepting to be uploaded
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'PNG', 'JPG'])
app.config['MAX_CONTENT_PATH'] = 4000000

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 3

global_model_data = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')
    
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS    
    
@app.route('/score', methods=['POST'])
def score():
    data = []
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('Please select a file')
            return redirect(request.referrer)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('Please select a file')
            return redirect(request.referrer)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            projected = project_face(file_path)
            if projected == True:
                data.append(file.filename)
                prepared_image = process_image(file_path[8:])
                prediction = model.predict(prepared_image)[0]
                score = 30
                for count, val in enumerate(prediction):
                    if count == 0:
                        score -= val * 0
                    elif count == 1:
                        score += val * 50
                    elif count == 2:
                        score += val * 70
                    elif count == 3:
                        score += val * 100
                    elif count == 4:
                        score += val * 200
                    elif count == 5:
                        score += val * 400
                print(score)        
                if score > 100:
                    score = 100
                score = round(score,0)
                print(prediction)
                data.append(score)
            else:
                flash('No faces were found in the image. Please try another')
                return redirect(request.referrer)      
    
    print(data)
    global_model_data = {'path': data[0], 'score': data[1]}
    with open('json_returns/model_return.json', 'w') as outfile:
        json.dump(global_model_data, outfile)
    return jsonify(data), 204

@app.route('/get_model_data')
def get_model_data():
    return jsonify(global_model_data)
    
    
@app.route('/projected_faces_web/<path:filename>')    
def download_file(filename):
    return send_from_directory('projected_faces_web', filename, as_attachment=True)
    

@app.route('/howitworks')
def howitworks():
    return render_template('howitworks.html')

@app.route('/perfection')
def perfection():
    return render_template('perfection.html')
    
@app.route('/pudding')
def pudding():
    return render_template('pudding.html')
    
@app.route('/test')
def test():
    return render_template('test.html')
    
@app.route('/endpoint')
def get_d3_data():
    df = pd.read_csv('data.csv') # Constructed however you need it
    return df.to_csv()
    


@app.route('/read_json')
def read_json():
    with open('json_returns/model_return.json') as data_file:    
        data = json.load(data_file)
    return jsonify(data)
    
@app.route('/wait', methods=["POST"])
def wait():
    time.sleep(3)
    return "", 204
    
@app.route('/basics', methods=["GET"])
def basics():
    return render_template('basics.html')
    
@app.route('/technical', methods=["GET"])
def technical():
    return render_template('technical.html')



if __name__ == '__main__':
    model = load_model('models/gpu_300_players_sigmoid_binary.h5')
    app.run(host='0.0.0.0', port=8080, debug=True)
