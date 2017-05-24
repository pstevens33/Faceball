<!-- Min: 3.066
Mean: 3.985
Max: 5.427

RMSE: 1.98
Accuracy: 1745/3662
Epochs: 10 -->

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
