'''
Code implements multi-perceptron neural network to classify MNIST images of
handwritten digits using Keras and Theano.  Based on code from
https://www.packtpub.com/books/content/training-neural-networks-efficiently-using-keras
Note: neural network geometry not optimized (accuracy could be much better!)
'''

from __future__ import division
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
import theano
from sklearn.cross_validation import train_test_split


X = np.load('../data/X.npy')
y = np.load('../data/y_wars.npy')

# X = X[:500]
# y = y[:500]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
y_train_ohe = np_utils.to_categorical(y_train)

input_shape = (128, 128, 3)


''' defines multi-layer-perceptron neural network '''
# available activation functions at:
# https://keras.io/activations/
# https://en.wikipedia.org/wiki/Activation_function
# options: 'linear', 'sigmoid', 'tanh', 'relu', 'softplus', 'softsign'
# there are other ways to initialize the weights besides 'uniform', too

model = Sequential() # sequence of layers
num_neurons_in_layer = 64 # number of neurons in a layer
num_inputs = X_train.shape[1] # number of features (784)
num_classes = y_train_ohe.shape[1]  # number of classes, 0-9

model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(input_dim=num_inputs,
                 output_dim=num_neurons_in_layer,
                 init='uniform',
                 activation='relu')) # only 12 neurons in this layer!
model.add(Dense(input_dim=num_neurons_in_layer,
                 output_dim=num_neurons_in_layer,
                 init='uniform',
                 activation='relu')) # only 12 neurons - keep softmax at last layer
model.add(Dense(input_dim=num_neurons_in_layer,
                 output_dim=num_neurons_in_layer,
                 init='uniform',
                 activation='relu')) # only 12 neurons - keep softmax at last layer
model.add(Dense(input_dim=num_neurons_in_layer,
                 output_dim=num_classes,
                 init='uniform',
                 activation='sigmoid')) # only 12 neurons - keep softmax at last layer
sgd = SGD(lr=0.001, decay=1e-7, momentum=0.95) # using stochastic gradient descent (keep)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"] ) # (keep)





# for i in np.arange(5,16,1):
#     model = define_nn_mlp_model(X_train, y_train_ohe, 200)
#     model.fit(X_train, y_train_ohe, nb_epoch=i, batch_size=100, verbose=1,
#               validation_split=0.1) # cross val to estimate test error
#     print_output(model, y_train, y_test, rng_seed)
#     if print_output(model, y_train, y_test, rng_seed) > max_acc:
#         max_acc = print_output(model, y_train, y_test, rng_seed)
#         max_batch = i

unique, counts = np.unique(y_train, return_counts=True)
class_count_dict = dict(zip(unique, counts))

class_weight = {0 : len(y_train) / (class_count_dict[0]),
                1 : len(y_train) / (class_count_dict[1]),
                2 : len(y_train) / (class_count_dict[2]),
                3 : len(y_train) / class_count_dict[3],
                4 : len(y_train) / class_count_dict[4],
                5 : len(y_train) / class_count_dict[5],
                6 : len(y_train) / (class_count_dict[6])}

model.fit(X_train, y_train_ohe, epochs=3, batch_size=16, verbose=1, validation_split=0.1) # cross val to estimate test error

predict = []
predict2 = model.predict(X_test, batch_size=16)
for i, temp in enumerate(predict2):
    x0 = predict2[i,0] * 1 * -0.3
    x1 = predict2[i,1] * 1 * 0.3
    x2 = predict2[i,2] * 2 * 1
    x3 = predict2[i,3] * 3 * 1
    x4 = predict2[i,4] * 4 * 12
    x5 = predict2[i,5] * 5 * 16
    x6 = predict2[i,6] * 6 * 20
    predict.append(round(sum([x0,x1,x2,x3,x4,x5,x6]),0))
predict = np.array(predict)
model_score = round(np.sqrt(np.mean(np.square(predict - y_test))), 2)

correct_guesses = 0
print(predict.min())
print(predict.mean())
print(predict.max())
print()
model_mean = predict.mean()


for i, image in enumerate(X_test):
    image = image.reshape((1,) + image.shape)
    predict3 = model.predict(image)
    for i, temp in enumerate(predict3):
        x0 = predict3[i,0] * 1 * -0.3
        x1 = predict3[i,1] * 1 * 0.3
        x2 = predict3[i,2] * 2 * 1
        x3 = predict3[i,3] * 3 * 1
        x4 = predict3[i,4] * 4 * 12
        x5 = predict3[i,5] * 5 * 16
        x6 = predict3[i,6] * 6 * 20
    predict_image = round(sum([x0,x1,x2,x3,x4,x5,x6]))
    if predict_image <= model_mean and y_test[i] < y_test.mean():
        correct_guesses += 1
    elif predict_image >= model_mean and y_test[i] > y_test.mean():
        correct_guesses += 1
print("Accuracy: {}/{}".format(correct_guesses, len(y_test)))
