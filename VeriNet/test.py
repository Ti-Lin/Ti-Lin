import os
from tensorflow.contrib.keras.api.keras.models import load_model
import tensorflow as tf

def fn(correct, predicted):
    return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                   logits=predicted)
dir_path = '/root/Ti-Lin-test/models/'
tf.set_random_seed(1215)
for dir_name in os.listdir(dir_path):
    keras_model = load_model(dir_path+dir_name, custom_objects={'fn': fn, 'tf': tf})
    for layer in keras_model.layers:
        print(dir(layer))