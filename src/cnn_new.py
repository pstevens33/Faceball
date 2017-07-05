'''
Code implements a convolutional neural network using keras and theano to scan images of faces of baseball players to predict their likeliness of success. (5 convolution layers, 3 fully-connected, output shape of 6)
'''

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.cross_validation import train_test_split

history = History()

# Load data from saved numpy arrays
X = np.load('../data/X_players.npy')
y = np.load('../data/y_players.npy')

# One hot encoding for multi-class labels
y_ohe = np_utils.to_categorical(y)

# Original Shape of Images
input_shape = (128, 128, 3)

''' defines multi-layer-perceptron neural network '''
model = Sequential() # sequence of layers
num_neurons_in_layer = 64 # number of neurons in a layer
num_inputs = X.shape[1] # number of features (784)
num_classes = y_ohe.shape[1]  # number of classes, 0-9

# 5 Convolution Layers (Dropout of 0.35 reduced overfitting and still allowed for adequate learning)
model.add(Conv2D(32, 3, 3, input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.35))

model.add(Conv2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.35))

model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.35))

model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.35))

model.add(Conv2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.35))

# Flatten to make proper dimensions
model.add(Flatten())

# 3 Fully-Connected layers
model.add(Dense(input_dim=128,
                 output_dim=num_neurons_in_layer,
                 init='uniform',
                 activation='relu'))
model.add(Dense(input_dim=num_neurons_in_layer,
                 output_dim=num_neurons_in_layer,
                 init='uniform',
                 activation='relu'))
model.add(Dense(input_dim=num_neurons_in_layer,
                 output_dim=num_classes,
                 init='uniform',
                 activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Necessary in order to plot loss and accuracy for each epoch
history = model.fit(X, y_ohe, epochs=300, batch_size=128, shuffle=True, verbose=1, validation_split=0.25) # 

# Plot history and loss during training
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('../data/plots/acc_300_binary_sigmoid_adam_0.35_3full.png')
plt.close()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('../data/plots/loss_300_binary_sigmoid_adam_0.35_3full.png')
plt.close()

model.save('../data/models/300_binary_sigmoid_adam_0.35_3full.h5')

