import os
from tensorflow.contrib.keras.api.keras.models import load_model
import tensorflow as tf
from train_resnet import ResidualStart, ResidualStart2
root_path = '/root/new-Ti-Lin/models/'
files = os.listdir(root_path)

def fn(correct, predicted):
    return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                   logits=predicted)
print(files)
for file_name in files:
    tf.set_random_seed(1215)
    if ('sigmoid' in file_name) or ('tanh' in file_name) or ('atan' in file_name):continue
    print(root_path+file_name)
    keras_model = load_model(root_path+file_name, custom_objects={'fn':fn, 'ResidualStart':ResidualStart, 'ResidualStart2':ResidualStart2, 'tf':tf})#custom_objects={'fn':fn, 'tf':tf})
    print(file_name, file=open("all_network_layer_.txt", "a"))
    for layer in keras_model.layers:
        print(type(layer), file=open("all_network_layer_.txt", "a"))








