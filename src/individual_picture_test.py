import numpy as np
import pandas as pd
from PIL import Image
from resizeimage import resizeimage

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

df = pd.read_pickle('../data/recognized_faces_df')

player = input('Enter name of an MLB pitcher: ')
df2 = df[df['name'] == player]
image_path = df2['image_path'].values

img1 = load_img('../data/projected_faces/' + image_path[0])
x = img_to_array(img1)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
X = np.array(x)
