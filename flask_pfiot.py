from flask import Flask, render_template, url_for, request
from flask_sqlalchemy import SQLAlchemy
import tensorflow as tf
import numpy as np

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

    # load data and process to tensor
    respeck_model_tflite.set_tensor(respeck_input, respeck_data)
    respeck_model_tflite.invoke()
    predicted_respeck = respeck_model_tflite.get_tensor(respeck_output)
    return np.argmax(predicted_respeck)


def thingy_inference(thingy_data):
    thingy_input = thingy_model_tflite.get_input_details()[0]['index']
    thingy_output = thingy_model_tflite.get_output_details()[0]['index']

    # load data and process to tensor
    thingy_model_tflite.set_tensor(thingy_input, thingy_data)
    thingy_model_tflite.invoke()
    # softmax
    predicted_thingy = thingy_model_tflite.get_tensor(thingy_output)
    return np.argmax(predicted_thingy)
###########################################################################


app = Flask(__name__)
app.config['JSON_AS_ASCII']=False
app.config['SECRET_KEY'] = '123456'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:root@localhost:3306/flaskdb'


@app.route('/',methods=['GET','POST'])
def home():
    return 'Hello'

@app.route('/inference',methods=['GET','POST'])
def nohome():
    # test
    randomnum = np.random.randn(1, 50, 6)
    randomnum = randomnum.astype('float32')
    return class_labels[respeck_inference(randomnum)]



if __name__ == '__main__':
    app.run(host='localhost', port="5000", debug=True)









