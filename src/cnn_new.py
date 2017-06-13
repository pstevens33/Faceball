'''
Code implements multi-perceptron neural network to classify MNIST images of
handwritten digits using Keras and Theano.  Based on code from
https://www.packtpub.com/books/content/training-neural-networks-efficiently-using-keras
Note: neural network geometry not optimized (accuracy could be much better!)
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
from keras.callbacks import History 
# import theano
from sklearn.cross_validation import train_test_split
# import os

# os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"

history = History()

X = np.load('../data/X_players.npy')
y = np.load('../data/y_players.npy')

# X = X[:500]
# y = y[:500]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)
y_ohe = np_utils.to_categorical(y)

input_shape = (128, 128, 3)


''' defines multi-layer-perceptron neural network '''
# available activation functions at:
# https://keras.io/activations/
# https://en.wikipedia.org/wiki/Activation_function
# options: 'linear', 'sigmoid', 'tanh', 'relu', 'softplus', 'softsign'
# there are other ways to initialize the weights besides 'uniform', too

model = Sequential() # sequence of layers
num_neurons_in_layer = 64 # number of neurons in a layer
num_inputs = X.shape[1] # number of features (784)
num_classes = y_ohe.shape[1]  # number of classes, 0-9

model.add(Conv2D(32, 3, 3, input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.20))

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


model.add(Flatten())

model.add(Dense(input_dim=128,
                 output_dim=num_classes,
                 init='uniform',
                 activation='sigmoid')) # only 12 neurons - keep softmax at last layer
sgd = SGD(lr=0.001, decay=1e-7, momentum=0.95) # using stochastic gradient descent (keep)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'] ) # (keep)


unique, counts = np.unique(y, return_counts=True)
class_count_dict = dict(zip(unique, counts))

history = model.fit(X, y_ohe, epochs=300, batch_size=128, shuffle=True, verbose=1, validation_split=0.25) # cross val to estimate test error


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('../data/plots/acc_300_binary_sigmoid_adam_0.35_1full.png')
plt.close()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('../data/plots/loss_300_binary_sigmoid_adam_0.35_1full.png')
plt.close()

model.save('../data/models/300_binary_sigmoid_adam_0.35_1full.h5')
