import os
import glob
import time
from PIL import Image
from os import listdir
from shutil import copyfile
from os.path import isfile, join
from matplotlib import pyplot as plt
# from keras import applications
# import keras_vggface
# from keras_vggface.vggface import VGGFace
# from keras.engine import Input
# from keras.models import Model
# import tensorflow as tf
# import numpy as np
# from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
# from keras_vggface.utils import preprocess_input
# from sklearn.metrics import classification_report, confusion_matrix
# from keras import backend as K
# import random
# from os import chdir as cd
# from keras.preprocessing.image import load_img
# from google.colab.patches import cv2_imshow
import cv2

# random.seed(42)
# tf.random.set_seed(42)

Height = 224
Width  = 224
BatchSize = 24
lr_rate=.0015
Version = 5
load_model = True
accuracy = 0
accuracyCount = 0
trainableCount = 30

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# def predict_vf(imagepath):
#     # TestPath  = '/content/drive/MyDrive/Kaggle-Autism/Autism-Data/test/autistic/001.jpg'


#     model_path = 'models/'
#     reconstructed_model = keras.models.load_model(model_path)
#     image = cv2.imread(imagepath)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     faces = faceCascade.detectMultiScale(
#             gray,
#             scaleFactor=1.3,
#             minNeighbors=3,
#             minSize=(30, 30)
#     )
#     crop = image.copy()
#     for (x, y, w, h) in faces:
#     #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         crop = image[y:y+h, x:x+w]
#         # cv2_imshow(crop)
#         # cv2.waitKey(0) 
#     # load an image from file
#     # image =  load_img(crop, target_size=(Height, Width))
#     crop = cv2.resize(crop, (224, 224))
#     image = keras.preprocessing.image.img_to_array(crop)
#     image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#     Y_pred = reconstructed_model.predict(image)
#     return Y_pred[0]

import numpy as np
import tflite_runtime.interpreter as tflite

def predict_vf(imagepath):
    
    # model_path = '/content/drive/MyDrive/Kaggle-Autism/models/h5/timestamp/'
    # reconstructed_model = keras.models.load_model(model_path)
    image = cv2.imread(imagepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30)
    )       
    crop = image.copy()                                        # detect only the facial region
    for (x, y, w, h) in faces:                      # crop the detected face
        crop = image[y:y+h, x:x+w]
        cv2.imshow('',crop)
        
    crop = cv2.resize(crop, (224, 224))             # resize image
    image = crop
    print(type(image))
    # image = keras.preprocessing.image.img_to_array(crop)    # convert image to array
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]),)      # reshape image
    # Y_pred = reconstructed_model.predict(image)             # predict probability of input image
    print(image.shape)
    
    # Load TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path="models/model.tflite")
    interpreter.allocate_tensors()

    image = image.astype('float32')

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on input data.
    input_shape = input_details[0]['shape']
    interpreter.set_tensor(input_details[0]['index'], image)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    Y_pred = interpreter.get_tensor(output_details[0]['index'])
    # print(output_data)
    return Y_pred[0]                                        # return autistic probability
