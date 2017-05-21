import numpy as np
from sklearn.cross_validation import train_test_split

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import RMSprop, Adam, Adadelta

# Grab data from pickles and split into train and test
X = np.load('../data/X.npy')
y = np.load('../data/y.npy')

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
input_shape = (128, 128, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='sigmoid'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='linear'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='relu'))

model.compile(optimizer='rmsprop',
              loss='mse')

#
# nb_filters = 8
# nb_conv = 5
#
# model = Sequential()
# model.add(Conv2D(nb_filters, nb_conv, nb_conv,
#                         border_mode='valid',
#                         input_shape=input_shape ) )
# model.add(Activation('relu'))
#
# model.add(Conv2D(nb_filters, nb_conv, nb_conv))
# model.add(Activation('relu'))
#
# model.add(Conv2D(nb_filters, nb_conv, nb_conv))
# model.add(Activation('relu'))
#
# model.add(Conv2D(nb_filters, nb_conv, nb_conv))
# model.add(Activation('relu'))
# model.add(Dropout(0.25))
#
# model.add(Conv2D(nb_filters*2, nb_conv, nb_conv))
# model.add(Activation('relu'))
#
# model.add(Conv2D(nb_filters*2, nb_conv, nb_conv))
# model.add(Activation('relu'))
#
# model.add(Conv2D(nb_filters*2, nb_conv, nb_conv))
# model.add(Activation('relu'))
#
# model.add(Conv2D(nb_filters*2, nb_conv, nb_conv))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
#
# model.add(Flatten())
# model.add(Dense(256))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
#
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
#
# model.add(Dense(1))
# model.add(Activation('linear'))
#
# model.compile(loss='mean_squared_error', optimizer=Adadelta())



batch_size = 16

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow(
        X_train,
        y_train,
        batch_size=batch_size)  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow(
        X_test,
        y_test,
        batch_size=batch_size)


model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)

# model.fit_generator(
#         train_generator,
#         steps_per_epoch=2000 // batch_size,
#         epochs=5)

predict = model.predict(X_test, batch_size=batch_size)
model_score = round(np.sqrt(np.mean(np.square(predict - y_test))), 2)


model.save_weights('weights/{}.h5'.format(model_score))  # always save your weights after training or during training

correct_guesses = 0
print(predict.min())
print(predict.mean())
print(predict.max())
print()
model_mean = predict.mean()
for i, image in enumerate(X):
    image = image.reshape((1,) + image.shape)
    if model.predict(image) < model_mean and y[i] < y.mean():
        correct_guesses += 1
    elif model.predict(image) > model_mean and y[i] > y.mean():
        correct_guesses += 1
print("Accuracy: {}/{}".format(correct_guesses, len(y)))
