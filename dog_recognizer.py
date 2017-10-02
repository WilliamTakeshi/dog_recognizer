import numpy as np
import tensorflow as tf
from glob import glob
import cv2                
from extract_bottleneck_features import *
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Sequential

from keras.preprocessing import image                  
from keras.layers import GlobalAveragePooling2D, Activation
from keras.layers import Dense
from keras.models import Sequential

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    global graph
    with graph.as_default():
        return np.argmax(ResNet50_model.predict(img))

# load list of dog names
f = open('dog_names.txt', 'r')
dog_names = [line for line in f.readlines()]
f.close()
#dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]
# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

### TODO: Obtain bottleneck features from another pre-trained CNN.
#bottleneck_features = np.load('bottleneck_features/DogResnet50Data.npz')

### TODO: Define your architecture.
Resnet50_model = Sequential()
Resnet50_model.add(GlobalAveragePooling2D(input_shape=(1,1,2048)))
Resnet50_model.add(Dense(133, activation='softmax'))

### TODO: Load the model weights with the best validation loss.
Resnet50_model.load_weights('saved_models/weights.best.Resnet50_2.hdf5')
graph = tf.get_default_graph()


### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def Resnet50_predict_breed(img_path):
    # extract bottleneck features
    global graph
    with graph.as_default():
        bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
        # obtain predicted vector
        predicted_vector = Resnet50_model.predict(bottleneck_feature)
        # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

### TODO: Write your algorithm.
### Feel free to use as many code cells as needed.
def dog_or_human(img_path):
    if dog_detector(img_path):
        return "dog"
    elif face_detector(img_path):
        return "human"
    else:
        return "neither"

def predict_breed(img_path):
    race = dog_or_human(img_path)
    if race == "neither":
        return race, 'Error'
    else:
        breed = Resnet50_predict_breed(img_path)
        return race, breed