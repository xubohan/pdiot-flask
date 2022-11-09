import numpy
from flask import Flask, render_template, url_for, request
# from flask_sqlalchemy import SQLAlchemy
import tensorflow as tf
import numpy as np
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
    return np.argmax(predicted_respeck)


def thingy_inference(thingy_data):
    thingy_input = thingy_model_tflite.get_input_details()[0]['index']
    thingy_output = thingy_model_tflite.get_output_details()[0]['index']
    thingy_data = thingy_data.astype('float32')
    # load data and process to tensor
    thingy_model_tflite.set_tensor(thingy_input, thingy_data)
    thingy_model_tflite.invoke()
    # softmax
    predicted_thingy = thingy_model_tflite.get_tensor(thingy_output)
    return np.argmax(predicted_thingy)
###########################################################################


app = Flask(__name__)
app.config['JSON_AS_ASCII']=False
# app.config['SECRET_KEY'] = '123456'
# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:root@localhost:3306/flaskdb'


@app.route('/',methods=['GET','POST'])
def index():
    return render_template('index.html')

# @app.route('/respeck',methods=['GET','POST'])
def respeck_pred(respeck_data):
    # test
    randomnum = np.random.randn(1, 50, 6)
    randomnum = randomnum.astype('float32')
    return class_labels[respeck_inference(randomnum)]

# @app.route('/thingy',methods=['GET','POST'])
def thingy_pred(thingy_data):
    # test
    randomnum = np.random.randn(1, 50, 9)
    randomnum = randomnum.astype('float32')
    return class_labels[thingy_inference(randomnum)]


@app.route('/post_data', methods=['POST'])
def receive_post():
    data = request.get_json()  # get JSON data
    data = data['freq']
    return data

# def preprocess(nv_data):

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
@app.route('/register', methods=['POST']) # enable GET method
def register():
    # print(request.stream.read())
    device_val = ''
    data_window = []
    # if request.method == "GET":
    #     comment = request.args.get("content")
    if request.method == "POST":
        if request.content_type.startswith('application/json'):
            device_val = request.json.get('device')
            data_window = request.json.get('dataWindow')
        elif request.content_type.startswith('multipart/form-data'):
            device_val = request.form.get('device')
            data_window = request.form.get('dataWindow')
        else:
            device_val = request.values.get("device")
            data_window = request.values.get('dataWindow')
    # wait for the real data...
    # data_window = numpy.array(data_window)
    if device_val == 'respeck':
        data_window = np.random.randn(1, 50, 6)
        return device_val + ' ' + class_labels[respeck_inference(data_window)]
    elif device_val == 'thingy':
        data_window = np.random.randn(1, 50, 9)
        return device_val + ' ' + class_labels[thingy_inference(data_window)]
    else:
        return 'welcome'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port="61000", debug=True)









