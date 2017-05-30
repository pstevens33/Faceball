from __future__ import division
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import theano
from sklearn.cross_validation import train_test_split


X = np.load('../data/X_batters.npy')
y = np.load('../data/y_batters.npy')

# X = X[:500]
# y = y[:500]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
y_train_ohe = np_utils.to_categorical(y_train)

input_shape = (128, 128, 3)
num_neurons_in_layer = 64 # number of neurons in a layer
num_inputs = X_train.shape[1] # number of features (784)
num_classes = y_train_ohe.shape[1]


model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=input_shape))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))

sgd = SGD(lr=0.001, decay=1e-7, momentum=0.95) # using stochastic gradient descent (keep)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"] ) # (keep)


model.fit(X_train, y_train_ohe, epochs=1, batch_size=16, verbose=1) # cross val to estimate test error

predict = []
predict2 = model.predict(X_test, batch_size=16)
for i, temp in enumerate(predict2):
    x0 = predict2[i,0] * 1 * -7.5
    x1 = predict2[i,1] * 1 * 0.5
    x2 = predict2[i,2] * 2 * 1
    x3 = predict2[i,3] * 3 * 10
    x4 = predict2[i,4] * 4 * 10
    x5 = predict2[i,5] * 5 * 15
    x6 = predict2[i,6] * 6 * 15
    x7 = predict2[i,7] * 7 * 15
    # x8 = predict2[i,8] * 8 * 1
    predict.append(round(sum([x0,x1,x2,x3,x4,x5,x6,x7]),0))
predict = np.array(predict)
model_score = round(np.sqrt(np.mean(np.square(predict - y_test))), 2)



model_mean = predict.mean()

correct_lower = 0
correct_higher = 0
correct_guesses = 0
guess0 = []
guess1 = []
guess2 = []
guess3 = []
guess4 = []
guess5 = []
guess6 = []
guess7 = []
guess8 = []

for i, image in enumerate(X_test):
    image = image.reshape((1,) + image.shape)
    predict3 = model.predict(image)
    x0 = predict3[0][0] * 1 * -7.5
    x1 = predict3[0][1] * 1 * 0.5
    x2 = predict3[0][2] * 2 * 1
    x3 = predict3[0][3] * 3 * 10
    x4 = predict3[0][4] * 4 * 10
    x5 = predict3[0][5] * 5 * 15
    x6 = predict3[0][6] * 6 * 15
    x7 = predict3[0][7] * 7 * 15
    # x8 = predict3[0][8] * 8 * 1
    predict_image = round(sum([x0,x1,x2,x3,x4,x5,x6,x7]))
    if predict_image <= model_mean and y_test[i] < y_test.mean():
        print("Lower...     Prediction: {}, Actual Y Value: {}".format(predict_image, y_test[i]))
        correct_lower += 1
    elif predict_image >= model_mean and y_test[i] > y_test.mean():
        print("Higher...    Prediction: {}, Actual Y Value: {}".format(predict_image, y_test[i]))
        correct_higher += 1
    else:
        print("Incorrect... Prediction: {}, Actual Y Value: {}".format(predict_image, y_test[i]))

    if y_test[i] == 0:
        guess0.append(predict_image)
    elif y_test[i] == 1:
        guess1.append(predict_image)
    elif y_test[i] == 2:
        guess2.append(predict_image)
    elif y_test[i] == 3:
        guess3.append(predict_image)
    elif y_test[i] == 4:
        guess4.append(predict_image)
    elif y_test[i] == 5:
        guess5.append(predict_image)
    elif y_test[i] == 6:
        guess6.append(predict_image)
    elif y_test[i] == 7:
        guess7.append(predict_image)
    elif y_test[i] == 8:
        guess8.append(predict_image)

print()
print(predict.min())
print(predict.mean())
print(predict.max())
print()

correct_guesses = correct_lower + correct_higher
print("Correct Lower: {}".format(correct_lower))
print("Correct Higher: {}".format(correct_higher))
print("Accuracy: {}/{}".format(correct_guesses, len(y_test)))
print()
print("0: Average Guess: {}".format(sum(guess0)/len(guess0)))
print("1: Average Guess: {}".format(sum(guess1)/len(guess1)))
print("2: Average Guess: {}".format(sum(guess2)/len(guess2)))
print("3: Average Guess: {}".format(sum(guess3)/len(guess3)))
print("4: Average Guess: {}".format(sum(guess4)/len(guess4)))
print("5+: Average Guess: {}".format(sum([sum(guess5),sum(guess6),sum(guess7),sum(guess8)])/sum([len(guess5),len(guess6),len(guess7),len(guess8)])))

model.save('models/vgg_model.h5')
