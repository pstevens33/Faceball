from flask import Flask, render_template, request
from flask import make_response
from functools import wraps, update_wrapper
from datetime import datetime
from os import listdir
from os.path import isfile, join


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 3


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    onlyfiles = [f for f in listdir('static/img/players') if isfile(join('static/img/players', f))]
    paths = []
    for i in onlyfiles:
        paths.append('img/players/' + i)
    test = [1,2,4]
    return render_template('predict.html', data=paths)

@app.route('/howitworks')
def howitworks():
    return render_template('howitworks.html')

@app.route('/perfection')
def perfection():
    return render_template('perfection.html')


@app.after_request
def add_header(response):
    response.cache_control.max_age = 300
    return response





if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
