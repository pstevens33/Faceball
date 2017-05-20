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

df = pd.read_json('../data/pitcher_stats_and_pics.json', lines=True)
image_paths = df['image_path'].values
eras = df['era'].values
# np.save('y', eras)


# Resize images to 85x128 and create numpy array of images as numpy arrays
# Only have to do this for loop once

# img1 = load_img('../data/test_pics/' + image_paths[0])
# x = img_to_array(img1)  # this is a Numpy array with shape (3, 150, 150)
# x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
# X = np.array(x)
#
# i = 0

# for image_path in image_paths[1:]:
#
#     with open('../data/' + image_path, 'r+b') as f:
#         with Image.open(f) as image:
#             cover = resizeimage.resize_cover(image, [85, 128])
#             cover.save('../data/test_pics/' + image_path, image.format)
#             print(i)
#             i += 1
#
#     img = load_img('../data/test_pics/' + image_path)  # this is a PIL image
#     x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
#     x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
#     X = np.concatenate((X,x))
#     print(X.shape)


# i = 0
# for batch in datagen.flow(x, batch_size=1,
#                           save_to_dir='preview', save_prefix='cat', save_format='jpeg'):
#     i += 1
#     if i > 10:
#         break
