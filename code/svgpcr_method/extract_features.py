import skimage.io as io
import os
from tqdm import tqdm
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input
import numpy as np

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, AveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.utils import to_categorical

input_tensor = Input(shape=(224, 224, 3))
vgg_model = VGG16(weights='imagenet',
                               include_top=False,
                               input_tensor=input_tensor)

#Adding custom Layers 
x = vgg_model.output
x = AveragePooling2D(pool_size=(7,7))(x)

# creating the final model 
model_final = Model(input = vgg_model.input, output = x)

# compile the model 
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

def extract_features(path, final_model, vgg_model):
    """
    Extract features from the images and save the labels and names. The pool512 is for having a 512 component vector.
    """
    images_list = []
    name_list = []
    label_list = []
    for root, subdirs, files in os.walk(path):
        for name in tqdm(files):
            label_list.append(int(root.split('/')[len(root.split('/'))-1]))
            image = io.imread(os.path.join(root,name))[:, :, :3]
            image = preprocess_input(image)
            images_list.append(image)
            name_list.append(name)



    images_list = np.array(images_list)
    name_list = np.array(name_list)
    label_list = np.array(label_list)
    label_list = to_categorical(label_list)[:,1:]
    features_vgg_pool512 = final_model.predict(images_list)
    features_vgg_pool512 = features_vgg_pool512.reshape(-1,512)
    features_vgg = vgg_model.predict(images_list, batch_size=8, verbose=1)
    
    return images_list, features_vgg, features_vgg_pool512, label_list, name_list
            
images_train, X_train, X_train_pool512, y_train, name_list_train = extract_features('../Train_non_experts_simple', model_final, vgg_model)
images_test, X_test, X_test_pool512, y_test, name_list_test = extract_features('../Test', model_final, vgg_model)

np.save("X_train", X_train)
np.save("X_test", X_test)
np.save("X_train_512pool", X_train_pool512)
np.save("X_test_512pool", X_test_pool512)
np.save("y_train", y_train)
np.save("y_test", y_test)
np.save("name_list_train", name_list_train)
np.save("name_list_test", name_list_test)

