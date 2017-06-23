import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.layers.core import K
from sklearn.cross_validation import train_test_split
import pandas as pd

history = load_model('../data/models/gpu_300_players_sigmoid_binary.h5')
print('Model loaded.')


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('../data/plots/lossplot_for_poster.png')
plt.close()



