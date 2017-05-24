import numpy as np
import pandas as pd
from PIL import Image
from resizeimage import resizeimage

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

df = pd.read_pickle('../data/recognized_faces_df')
image_paths = df['image_path'].values
eras = df['era'].values
# np.save('y', eras)



# Resize images to 85x128 and create numpy array of images as numpy arrays
# Only have to do this for loop once

img1 = load_img('../data/projected_faces/' + image_paths[0])
x = img_to_array(img1)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
X = np.array(x)

i = 0


### By initializing the numpy array with a specific size and adding data by indexing into it, you save a ton of time but the file is twice as big as concatenation
### Append to an ordinary python list, it is way faster than concatenation and is the same size

temp = np.load('../data/X.npy')
temp.shape
new_x = []

for i,image_path in enumerate(image_paths):

    img = load_img('../data/projected_faces/' + image_path)  # this is a PIL image
    temp_x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    temp_x.tolist()
    # x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
    new_x.append(temp_x)
    print(i)


# i = 0
# for batch in datagen.flow(x, batch_size=1,
#                           save_to_dir='preview', save_prefix='cat', save_format='jpeg'):
#     i += 1
#     if i > 10:
#         break
