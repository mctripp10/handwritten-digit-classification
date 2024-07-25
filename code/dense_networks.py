from tensorflow import keras
from keras import optimizers
from keras import activations
from keras.callbacks import EarlyStopping
from keras.callbacks import CallbackList
from pprint import pprint
import numpy as np
import tensorflow as tf

'''
Michael Tripp
CS614: Machine Learning
Lab 3: Neural Networks
3/22/2023

Program to simulate dense neural networks using Keras
'''

def extract_file_data(fname):
    file = open(fname)
    data = []
    labels = []
    labels10 = []
    
    line = file.readline().strip()

    while line:
        line = np.array(line.split(",")).astype(float)
        data.append(line[:-1])
        labels.append(line[-1])
        line = file.readline().strip()
        
    file.close()
    data = np.array(data)
    labels = np.array(labels)
    
    return data, labels

''' Setup '''

filepath = "./data"      # Insert training data file path here
fname = filepath + "optdigits.tra"
train_images, train_labels = extract_file_data(fname)
train_labels = keras.utils.to_categorical(train_labels, 10)

fname = filepath + "optdigits.tes"
test_images, test_labels = extract_file_data(fname)
test_labels = keras.utils.to_categorical(test_labels, 10)

''' Parameters/hyperparameters'''

hiddenLayers = 3
layerSize = 32
activationFunction = 'relu'
lossFunction = 'categorical_crossentropy'
learningRate = 0.001
momentum = 0.8
numEpochs = 50
batch_size_val = 50

''' Build Model '''

model = keras.Sequential()
model.add(keras.layers.Dense(64, input_dim=64))
for i in range(hiddenLayers):
      model.add(keras.layers.Dense(layerSize,
                                  activation=activationFunction))
model.add(keras.layers.Dense(10, activation='softmax'))
sgd = optimizers.SGD(learningRate, momentum, nesterov=False)
model.compile(loss=lossFunction, optimizer=sgd, 
              metrics=['accuracy'])
stoppingEpoch = EarlyStopping(monitor='val_loss', 
                              mode='min', verbose=1, patience=10)
model.fit(train_images, train_labels, validation_split = 0.2,
          callbacks = [stoppingEpoch], epochs=numEpochs, batch_size = batch_size_val,
          verbose = 1)

''' Save and Test Model '''

modelName = "test_model.h5"
filepath = "insert file path here"          # Insert model file path here
model.save(f"{filepath}{modelName}")

currModel = keras.models.load_model(f"C:/Users/Admin/Documents/College/Spring 2023!!/CS614/lab3/{modelName}")
_, trainAccuracy = currModel.evaluate(train_images, train_labels, verbose=1)
_, testAccuracy = currModel.evaluate(test_images, test_labels, verbose=1)