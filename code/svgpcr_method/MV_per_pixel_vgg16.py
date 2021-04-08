
import os
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import skimage.io as io

import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
#prevent tensorflow from allocating the entire GPU memory at once
config = tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)

config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


import keras
from keras import optimizers
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, AveragePooling2D
from keras import backend as k
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.utils import to_categorical


def read_images(path):
    images_list = []
    label_list = []
    for root, subdirs, files in os.walk(path):
        for name in tqdm(files):
            label_list.append(int(root.split('/')[len(root.split('/'))-1]))
            image = io.imread(os.path.join(root,name))[:, :, :3]
            image = preprocess_input(image)
            images_list.append(image)

    images_list = np.array(images_list)
    label_list = np.array(label_list)
    label_list = to_categorical(label_list)[:,1:]

    return images_list, label_list

#Results_folder
if not os.path.exists('./Results/'):
    os.makedirs('./Results/')

#Loading data
X_train, _ = read_images('../Train_non_experts_simple')
X_test, y_test = read_images('../Test')

print('size', X_train.shape)
indexing=np.load("./features_pretrained/indexing.npy")
X_train=X_train[indexing]
y_train=np.load("./features_pretrained/y_train_MV_per_pixel.npy")

X_train_sub, X_valid, y_train_sub, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

input_tensor = Input(shape=(224, 224, 3))
vgg_model = VGG16(weights='imagenet',
                               include_top=False,
                               input_tensor=input_tensor)

# Freeze n number of layers from the last
for layer in vgg_model.layers: layer.trainable = False

#Adding custom Layers
x = vgg_model.output
x = Flatten()(x)
x = Dense(4096, activation='relu')(x)
x = Dense(4096, activation='relu')(x)
x = Dense(1000, activation='relu')(x)
x = Dense(3, activation='softmax')(x)

# creating the final model
model = Model(input = vgg_model.input, output = x)

print(model.summary())
print(y_train_sub.shape, y_test.shape,y_valid.shape)

# Check the trainable status of the individual layers
for layer in model.layers: print(layer, layer.trainable)

sgd = optimizers.SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
adam = optimizers.adam(lr=1e-5)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

datagen = ImageDataGenerator(
    featurewise_center=True,
    rotation_range=40,
    width_shift_range=0.4,
    height_shift_range=0.3,
    horizontal_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(X_train_sub)

# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(X_train_sub, y_train_sub, batch_size=16), validation_data=(X_valid, y_valid),
                    steps_per_epoch=len(X_train_sub) / 16, epochs=50, verbose=1)

probs = model.predict(X_test)
preds = np.argmax(probs,axis=1)

np.save("probs_MV_per_pixel", probs)

clas_report = classification_report(np.argmax(y_test,1), preds, digits=4)
roc_macro = roc_auc_score(y_test, probs, average='macro')

with open('./Results/results_MV_per_pixel_vgg_data_aug.txt', 'w') as the_file:
    the_file.write(clas_report)
    the_file.write('\n\nAUC   ' + str(roc_macro))


# serialize model to JSON
model_json = model.to_json()
with open("vgg16_MV_perpixel.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("vgg16_MV_perpixel.h5")
print("Saved model to disk")
