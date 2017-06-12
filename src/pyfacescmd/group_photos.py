import pandas as pd
from PIL import Image
import cv2


df_batters = pd.read_pickle('../../data/recognized_faces_batters_df')
df_pitchers = pd.read_pickle('../../data/recognized_faces_pitchers_df')
df = pd.concat([df_batters, df_pitchers])
df = df.reset_index()
df['avg_war'] = round(df['war'] / df['years_of_service'],0)
for i in range(df.shape[0]):
    if df.loc[i, 'avg_war'] < 0:
        df.loc[i, 'avg_war'] = 0
    elif df.loc[i, 'avg_war'] > 5:
        df.loc[i, 'avg_war'] = 5
image_paths = df['image_path'].values
wars = df['war'].values
avg_wars = df['avg_war'].values



# Resize images to 85x128 and create numpy array of images as numpy arrays
# Only have to do this for loop once


### By initializing the numpy array with a specific size and adding data by indexing into it, you save a ton of time but the file is twice as big as concatenation
### Append to an ordinary python list, it is way faster than concatenation and is the same size



for i,image_path in enumerate(image_paths):

    image = cv2.imread('../../data/projected_faces/' + image_path)
    if df.loc[i,'avg_war'] == 0:
        cv2.imwrite("../../data/photos_by_war/war_0/{}".format(image_path[8:]), image)
    elif df.loc[i,'avg_war'] == 1:
        cv2.imwrite("../../data/photos_by_war/war_1/{}".format(image_path[8:]), image)
    elif df.loc[i,'avg_war'] == 2:
        cv2.imwrite("../../data/photos_by_war/war_2/{}".format(image_path[8:]), image)
    elif df.loc[i,'avg_war'] == 3:
        cv2.imwrite("../../data/photos_by_war/war_3/{}".format(image_path[8:]), image)
    elif df.loc[i,'avg_war'] == 4:
        cv2.imwrite("../../data/photos_by_war/war_4/{}".format(image_path[8:]), image)
    elif df.loc[i,'avg_war'] == 5:
        cv2.imwrite("../../data/photos_by_war/war_5/{}".format(image_path[8:]), image)
    
    print(i)

