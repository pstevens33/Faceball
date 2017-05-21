Min: 3.066
Mean: 3.985
Max: 5.427

RMSE: 1.98
Accuracy: 1745/3662
Epochs: 10

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
