import numpy as np
import pandas as pd
from PIL import Image
from resizeimage import resizeimage
# from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img



def process_image(image_path):

    img = load_img('projected_faces_web/' + image_path)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x.tolist()
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
    X = np.array(x)
    return(X)


