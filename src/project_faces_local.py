import numpy as np
import pandas as pd
import sys
import dlib
import cv2
import openface

def project_face(path):

    # You can download the required pre-trained face detection model here:
    # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    predictor_model = "../face-detection-models/shape_predictor_68_face_landmarks.dat"


    # Create a HOG face detector using the built-in dlib class
    face_detector = dlib.get_frontal_face_detector()
    face_pose_predictor = dlib.shape_predictor(predictor_model)
    face_aligner = openface.AlignDlib(predictor_model)



    file_name = path

    # Load the image
    image = cv2.imread(file_name)
    # if image != None:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Run the HOG face detector on the image data
    detected_faces = face_detector(image, 1)

    # print("Found {} faces in image# {}".format(len(detected_faces), count))

    # Loop through each face we found in the image
    file_written = False
    for i, face_rect in enumerate(detected_faces):
        pose_landmarks = face_pose_predictor(image, face_rect)
        alignedFace = face_aligner.align(128, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        cv2.imwrite("projected_faces_web/{}".format(file_name[8:]), alignedFace)
        return True
    
    return False





