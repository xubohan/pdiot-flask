import tensorflow as tf
import numpy as np

# respect 50*6
# thingy  50*9


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

import time
if __name__ == '__main__':
    for i in range(10):
        randomnum = np.random.randn(1, 50, 6)
        randomnum = randomnum.astype('float32')
        t0 = time.time()
        print(respeck_inference(randomnum), time.time()-t0)














