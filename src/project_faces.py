import numpy as np
import pandas as pd
import sys
import dlib
import cv2
import openface

# You can download the required pre-trained face detection model here:
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor_model = "../face-detection-models/shape_predictor_68_face_landmarks.dat"


# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)
face_aligner = openface.AlignDlib(predictor_model)

df = pd.read_json('../data/batter_pics.json', lines=True)
image_paths = df['image_path'].values
wars = df['war'].values
names = df['name'].values
years_of_service = df['years_of_service'].values
image_paths_to_pickle = []
wars_to_pickle = []
names_to_pickle = []
years_to_pickle = []


for count, image_path in enumerate(image_paths):
    image = None
    file_name = '../data/' + image_path

    # Load the image
    image = cv2.imread(file_name)
    # if image != None:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Run the HOG face detector on the image data
    detected_faces = face_detector(image, 1)

    print("Found {} faces in image# {}".format(len(detected_faces), count))

    if len(detected_faces) != 0:
        image_paths_to_pickle.append(image_paths[count])
        wars_to_pickle.append(wars[count])
        names_to_pickle.append(names[count])
        years_to_pickle.append(years_of_service[count])

    # Loop through each face we found in the image
    for i, face_rect in enumerate(detected_faces):

    	# Detected faces are returned as an object with the coordinates
    	# of the top, left, right and bottom edges
    	#print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))

    	# Get the the face's pose
    	pose_landmarks = face_pose_predictor(image, face_rect)

    	# Use openface to calculate and perform the face alignment
    	alignedFace = face_aligner.align(128, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

    	# Save the aligned image to a file
    	cv2.imwrite("../data/projected_faces/{}".format(file_name[8:]), alignedFace)




df2 = pd.DataFrame()
df2['name'] = pd.Series(names_to_pickle)
df2['image_path'] = pd.Series(image_paths_to_pickle)
df2['war'] = pd.Series(wars_to_pickle)
df2['years_of_service'] = pd.Series(years_to_pickle)
df2.to_pickle('../data/recognized_faces_batters_df')
