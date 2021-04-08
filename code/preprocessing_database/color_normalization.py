import os
import numpy as np
import cv2
import pandas as pd
import histomicstk as htk

save_core = "../core_set_normalized/"
save_evaluation = "../eval_set_normalized/"
if not os.path.exists(save_core):
    os.makedirs(save_core)
if not os.path.exists(save_evaluation):
    os.makedirs(save_evaluation)

core_path = '../images/'
evaluation_path = '../evaluation_set_RGB/'

core_directory = os.listdir(core_path)
evaluation_directory=os.listdir(evaluation_path)


#Reference image
im_path_ref = core_path + core_directory[0]
im_RGB = cv2.imread(im_path_ref)[:, :, :3]
meanRef, stdRef = htk.preprocessing.color_conversion.lab_mean_std(im_RGB)

#Normalizing evaluation ROIs
for im in evaluation_directory:
    print("Reading image ", im)
    #Paths of the image
    im_path = evaluation_path + im
    #Reading mask and image
    im_RGB = cv2.imread(im_path)[:, :, :3]
    imNmzd = htk.preprocessing.color_normalization.reinhard(im_RGB, meanRef, stdRef)

    #save normalized
    save_path = save_evaluation+im
    cv2.imwrite(save_path, imNmzd)

#Normalizing core ROIs
for im in core_directory:
    print("Reading image ", im)
    #Paths of the image
    im_path = core_path + im
    #Reading mask and image
    im_RGB = cv2.imread(im_path)[:, :, :3]
    imNmzd = htk.preprocessing.color_normalization.reinhard(im_RGB, meanRef, stdRef)

    #save normalized
    save_path = save_core+im
    cv2.imwrite(save_path, imNmzd)
