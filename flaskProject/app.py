import numpy
from flask import Flask, render_template, url_for, request, redirect
# from flask_sqlalchemy import SQLAlchemy
# import tensorflow as tf
import numpy as np
import tensorflow as tf
import firebase_pdiot as fp

# import pandas as pd

# respect 50*6
# thingy  50*9

############################################### class table
class_labels = [
    'Desk work',
    'Walking at normal speed',
    'Climbing stairs',
    'Descending stairs',
    'Sitting',
    'Sitting bent forward',
    'Sitting bent backward',
    'Standing',
    'Running',
    'Lying down left',
    'Lying down right',
    'Lying down on back',
    'Lying down on stomach',
    'Movement',
]
############################################### inference
respeck_model_tflite = tf.lite.Interpreter("respeck_model.tflite")
thingy_model_tflite = tf.lite.Interpreter("thingy_model.tflite")

respeck_model_tflite.allocate_tensors()
thingy_model_tflite.allocate_tensors()


def respeck_inference(respeck_data):
    respeck_input = respeck_model_tflite.get_input_details()[0]['index']
    respeck_output = respeck_model_tflite.get_output_details()[0]['index']
    respeck_data = respeck_data.astype('float32')
    # load data and process to tensor
    respeck_model_tflite.set_tensor(respeck_input, respeck_data)
    respeck_model_tflite.invoke()
    predicted_respeck = respeck_model_tflite.get_tensor(respeck_output)
    return predicted_respeck


def thingy_inference(thingy_data):
    thingy_input = thingy_model_tflite.get_input_details()[0]['index']
    thingy_output = thingy_model_tflite.get_output_details()[0]['index']
    thingy_data = thingy_data.astype('float32')
    # load data and process to tensor
    thingy_model_tflite.set_tensor(thingy_input, thingy_data)
    thingy_model_tflite.invoke()
    # softmax
    predicted_thingy = thingy_model_tflite.get_tensor(thingy_output)
    return predicted_thingy


###########################################################################


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


# app.config['SECRET_KEY'] = '123456'
# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:root@localhost:3306/flaskdb'


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


# @app.route('/respeck',methods=['GET','POST'])
def respeck_pred(respeck_data):
    # test
    return np.argmax(respeck_inference(respeck_data))


# @app.route('/thingy',methods=['GET','POST'])
def thingy_pred(thingy_data):
    # test
    return np.argmax(thingy_inference(thingy_data))


''' test code js
var url = "http://172.16.3.155:61000/register";
var params = {device: 'respeck', dataWindow: [Math.random()]};
var xhr = new XMLHttpRequest();
xhr.open("POST", url, true);
xhr.setRequestHeader("Content-Type", "application/json");
xhr.onload = function (e) {
  if (xhr.readyState === 4) {
    if (xhr.status === 200) {
      console.log(xhr.responseText);
    } else {
      console.error(xhr.statusText);
    }
  }
};
xhr.onerror = function (e) {
  console.error(xhr.statusText);
};
xhr.send(JSON.stringify(params));
'''


@app.route('/register', methods=['POST', 'GET'])  # enable GET method
def register():
    if request.method == "GET":
        # comment = request.args.get("content")
        return redirect(url_for('index'))
    if request.method == 'POST':
        if request.content_type.startswith('application/json'):
            username = request.json.get('username')
            password = request.json.get('password')
            if not fp.create_account_to_db(username, password):
                return '0'
    return '1'


@app.route('/login', methods=['POST'])
def login():
    if request.method == "GET":
        # comment = request.args.get("content")
        return redirect(url_for('index'))
    if request.method == 'POST':
        if request.content_type.startswith('application/json'):
            username = request.json.get('username')
            password = request.json.get('password')
            return str(fp.check_username_password(username, password))


@app.route('/inference', methods=['POST', 'GET'])  # enable GET method
def inference():
    device_val = ''
    data_window = []
    username = ''
    if request.method == "GET":
        return redirect(url_for('index'))
    if request.method == "POST":
        if request.content_type.startswith('application/json'):
            # device_val = request.get_json()
            username = request.json.get('username')
            device_val = request.json.get('device')
            data_window = request.json.get('dataWindow')
        # elif request.content_type.startswith('multipart/form-data'):
        #     device_val = request.form.get('device')
        #     data_window = request.form.get('dataWindow')
        # else:
        #     device_val = request.values.get("device")
        #     data_window = request.values.get('dataWindow')
    # wait for the real data...
    # data_window = numpy.array(data_window)
    if device_val == 'respeck':
        # np_dw = np.ndarray(data_window)
        # np_dw.resize(1, 50, 6)
        np_dw = np.random.randn(1, 50, 6)
        outcome = respeck_pred(np_dw)
        outcome = int(outcome)
        fp.upload_data(username, device_val, data_window, outcome)
        return class_labels[outcome]
    elif device_val == 'thingy':
        # np_dw = np.ndarray(data_window)
        # np_dw.resize(1, 50, 9)
        np_dw = np.random.randn(1, 50, 9)
        outcome = thingy_pred(np_dw)
        outcome = int(outcome)
        fp.upload_data(username, device_val, data_window, outcome)
        return class_labels[outcome]
    elif device_val == 'both':
        flag = data_window[0].size == 300
        data_thingy = data_window[flag ^ 1]
        np_dw_thingy = np.ndarray(data_thingy)
        np_dw_thingy.resize((1, 50, 9))
        outcome_thingy = thingy_pred(np_dw_thingy)
        outcome_thingy = int(outcome_thingy)
        #fp.upload_data(username, 'thingy', data_thingy, outcome_thingy)

        data_respeck = data_window[flag ^ 1]
        np_dw_respeck = np.ndarray(data_respeck)
        np_dw_respeck.resize((1, 50, 6))
        outcome_respeck = respeck_pred(np_dw_thingy)
        outcome_respeck = int(outcome_respeck)
        #fp.upload_data(username, 'respeck', data_respeck, outcome_respeck)
        # waiting

    else:
        return '0'


@app.route('/test', methods=['POST', 'GET'])
def test_interface():
    if request.method == "GET":
        return 'test page'
    fp.test(request.get_json())
    return 'test data adds to database'


@app.route('/history', methods=['POST', 'GET'])
def history_data():
    if request.method == "GET":
        #     comment = request.args.get("content")
        return redirect(url_for('index'))
    if request.method == "POST":
        if request.content_type.startswith('application/json'):
            # device_val = request.get_json()
            username = request.json.get('username')
            return str(fp.history_classification(username))



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
