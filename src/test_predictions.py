import numpy as np
from keras.models import load_model
from keras.layers.core import K
from sklearn.cross_validation import train_test_split


X = np.load('../data/X_players.npy')
y = np.load('../data/y_players.npy')



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = load_model('../data/models/gpu_300_players_sigmoid_binary.h5')
print('Model loaded.')

predict = model.predict_classes(X_test)
zipped = list(zip(y_test,predict))

unique, counts = np.unique(y_test, return_counts=True)
class_count_dict = dict(zip(unique, counts))
print(class_count_dict)

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
    print(i)
    image = image.reshape((1,) + image.shape)
    predict_image = model.predict_classes(image)[0]
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


print("0: Average Guess: {}".format(sum(guess0)/len(guess0)))
print("1: Average Guess: {}".format(sum(guess1)/len(guess1)))
print("2: Average Guess: {}".format(sum(guess2)/len(guess2)))
print("3: Average Guess: {}".format(sum(guess3)/len(guess3)))
print("4: Average Guess: {}".format(sum(guess4)/len(guess4)))
print("5+: Average Guess: {}".format(sum([sum(guess5),sum(guess6),sum(guess7),sum(guess8)])/sum([len(guess5),len(guess6),len(guess7),len(guess8)])))
