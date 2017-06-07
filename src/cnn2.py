'''
Code implements multi-perceptron neural network to classify MNIST images of
handwritten digits using Keras and Theano.  Based on code from
https://www.packtpub.com/books/content/training-neural-networks-efficiently-using-keras
Note: neural network geometry not optimized (accuracy could be much better!)
'''

from __future__ import division
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
# import theano
from sklearn.cross_validation import train_test_split
# import os

# os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"


X = np.load('../data/X_players.npy')
y = np.load('../data/y_players.npy')

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

model.add(Conv2D(32, 3, 3, input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(input_dim=128,
                 output_dim=num_neurons_in_layer,
                 init='uniform',
                 activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(input_dim=num_neurons_in_layer,
                 output_dim=num_neurons_in_layer,
                 init='uniform',
                 activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(input_dim=num_neurons_in_layer,
                 output_dim=num_neurons_in_layer,
                 init='uniform',
                 activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(input_dim=num_neurons_in_layer,
                 output_dim=num_classes,
                 init='uniform',
                 activation='relu')) # only 12 neurons - keep softmax at last layer
sgd = SGD(lr=0.001, decay=1e-7, momentum=0.95) # using stochastic gradient descent (keep)
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'] ) # (keep)


# batch_size = 16

# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True)
#
# train_generator = train_datagen.flow(
#         X_train,
#         y_train,
#         batch_size=batch_size)


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

# class_weight = {0 : len(y_train) / (class_count_dict[0]),
#                 1 : len(y_train) / (class_count_dict[1]),
#                 2 : len(y_train) / (class_count_dict[2]),
#                 3 : len(y_train) / class_count_dict[3],
#                 4 : len(y_train) / class_count_dict[4],
#                 5 : len(y_train) / class_count_dict[5],
#                 6 : len(y_train) / (class_count_dict[6]),
#                 7 : len(y_train) / (class_count_dict[7])}



model.fit(X_train, y_train_ohe, epochs=500, batch_size=128, verbose=1) # cross val to estimate test error
predict = model.predict_classes(X_test, batch_size=64)
unique, counts = np.unique(predict, return_counts=True)
class_count_dict = dict(zip(unique, counts))
print(class_count_dict)

# predict = []
predict2 = model.predict(X_test, batch_size=64)
# for i, temp in enumerate(predict2):
#     x0 = predict2[i,0] * 1 * -7.5
#     x1 = predict2[i,1] * 1 * 0.5
#     x2 = predict2[i,2] * 2 * 1
#     x3 = predict2[i,3] * 3 * 10
#     x4 = predict2[i,4] * 4 * 10
#     x5 = predict2[i,5] * 5 * 15
#     x6 = predict2[i,6] * 6 * 15
#     x7 = predict2[i,7] * 7 * 15
#     # x8 = predict2[i,8] * 8 * 1
#     predict.append(round(sum([x0,x1,x2,x3,x4,x5,x6,x7]),0))
# predict = np.array(predict)
# model_score = round(np.sqrt(np.mean(np.square(predict - y_test))), 2)
#
#
#
# model_mean = predict.mean()
#
# correct_lower = 0
# correct_higher = 0
# correct_guesses = 0
# guess0 = []
# guess1 = []
# guess2 = []
# guess3 = []
# guess4 = []
# guess5 = []
# guess6 = []
# guess7 = []
# guess8 = []
#
# for i, image in enumerate(X_test):
#     image = image.reshape((1,) + image.shape)
#     predict3 = model.predict(image)
#     x0 = predict3[0][0] * 1 * -7.5
#     x1 = predict3[0][1] * 1 * 0.5
#     x2 = predict3[0][2] * 2 * 1
#     x3 = predict3[0][3] * 3 * 10
#     x4 = predict3[0][4] * 4 * 10
#     x5 = predict3[0][5] * 5 * 15
#     x6 = predict3[0][6] * 6 * 15
#     x7 = predict3[0][7] * 7 * 15
#     # x8 = predict3[0][8] * 8 * 1
#     predict_image = round(sum([x0,x1,x2,x3,x4,x5,x6,x7]))
#     if predict_image <= model_mean and y_test[i] < y_test.mean():
#         print("Lower...     Prediction: {}, Actual Y Value: {}".format(predict_image, y_test[i]))
#         correct_lower += 1
#     elif predict_image >= model_mean and y_test[i] > y_test.mean():
#         print("Higher...    Prediction: {}, Actual Y Value: {}".format(predict_image, y_test[i]))
#         correct_higher += 1
#     else:
#         print("Incorrect... Prediction: {}, Actual Y Value: {}".format(predict_image, y_test[i]))
#
#     if y_test[i] == 0:
#         guess0.append(predict_image)
#     elif y_test[i] == 1:
#         guess1.append(predict_image)
#     elif y_test[i] == 2:
#         guess2.append(predict_image)
#     elif y_test[i] == 3:
#         guess3.append(predict_image)
#     elif y_test[i] == 4:
#         guess4.append(predict_image)
#     elif y_test[i] == 5:
#         guess5.append(predict_image)
#     elif y_test[i] == 6:
#         guess6.append(predict_image)
#     elif y_test[i] == 7:
#         guess7.append(predict_image)
#     elif y_test[i] == 8:
#         guess8.append(predict_image)
#
# print()
# print(predict.min())
# print(predict.mean())
# print(predict.max())
# print()
#
# correct_guesses = correct_lower + correct_higher
# print("Correct Lower: {}".format(correct_lower))
# print("Correct Higher: {}".format(correct_higher))
# print("Accuracy: {}/{}".format(correct_guesses, len(y_test)))
# print()
# print("0: Average Guess: {}".format(sum(guess0)/len(guess0)))
# print("1: Average Guess: {}".format(sum(guess1)/len(guess1)))
# print("2: Average Guess: {}".format(sum(guess2)/len(guess2)))
# print("3: Average Guess: {}".format(sum(guess3)/len(guess3)))
# print("4: Average Guess: {}".format(sum(guess4)/len(guess4)))
# print("5+: Average Guess: {}".format(sum([sum(guess5),sum(guess6),sum(guess7),sum(guess8)])/sum([len(guess5),len(guess6),len(guess7),len(guess8)])))
#

model.save('../data/models/gpu_500_players.h5')
