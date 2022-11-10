import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.datasets import mnist,cifar10
from tensorflow.contrib.keras.api.keras.layers import Lambda
import matplotlib.pyplot as plt 
import numpy as np
import random

def train(data, file_name, activation=tf.nn.relu):
    '''
    mnist input_shape=(28,28);epochs=5
    cifa input_shape=(32,32,3);epochs=100
    '''
    model = Sequential()
    model.add(tf.keras.layers.Flatten(input_shape = (32, 32, 3)))
    #model.add(tf.keras.layers.Dense(128, activation = tf.atan))
    model.add(tf.keras.layers.Dense(128))
    model.add(Lambda(activation))
    #model.add(tf.keras.layers.Dense(128))
    #model.add(Lambda(activation))
    #model.add(tf.keras.layers.Dense(128))
    #model.add(Lambda(activation))
    #model.add(tf.keras.layers.Dense(128))
    #model.add(Lambda(activation))
    model.add(tf.keras.layers.Dense(10, activation = 'softmax'))

    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ["accuracy"])
    model.fit(data[0], data[1], epochs = 100) 

    test_loss, test_acc = model.evaluate(data[2], data[3])
    print("Loss = %.3f" % test_loss)
    print("Accuracy = %.3f" % test_acc)

    n = random.randint(0, 9999)
    '''
    plt.imshow(x_test[n])
    plt.show()
    '''
    prediction = model.predict(x_test)
    print("The handwritten number in the image is %d" % np.argmax(prediction[n]))
    model.save(file_name)
if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()#mnist.load_data()
    x_train = x_train / 255
    x_test = x_test / 255
    data = [x_train, y_train, x_test, y_test]
    train(data, 'cifa_fnn_3layer_256_sigmoid',tf.sigmoid)
    train(data, 'cifa_fnn_3layer_256_tanh',tf.tanh)
    train(data, 'cifa_fnn_3layer_256_atan',tf.atan)
    
