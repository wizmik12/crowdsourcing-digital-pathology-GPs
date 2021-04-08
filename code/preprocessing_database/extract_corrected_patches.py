import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import skimage.io as io
import pandas as pd
import tqdm as tqdm
from extracting_patches import labeling, label_image, sliding_window

if __name__ == "__main__":

    dir2Save = "../patches_core/"
    if not os.path.exists(dir2Save):
        os.makedirs(dir2Save)

    images_path = '../core_set_normalized/'
    #images_path = './evaluation_set_RGB/'
    images_directory = os.listdir(images_path)
    mask_path = '../masks/'
    #mask_path = './eval_masks_to_train_model/SP/'
    mask_directory = os.listdir(mask_path)


    evaluation_set_patches = pd.DataFrame({'im':[],'patch':[],'name': [], 'institute': [], 'label':[], 'x':[], 'y':[]})
    actual_im = 0
    for im in images_directory:
        actual_im += 1
        print("Reading image ", im, " ", actual_im,"/151")
        #Paths of the image and mask
        im_path = images_path + im
        institute, name_im = im.split('-')[1], im.split('-')[2]
        index_mask=np.argmax([name_im in x for x in mask_directory])
        mask = mask_directory[index_mask]
        im_mask = mask_path + mask
        #Reading mask and image
        im_RGB = io.imread(im_path)[:, :, :3]
        annotation_mask = io.imread(im_mask)
        #Labeling the image
        mask_label = labeling(annotation_mask)

        for (x, y, window) in sliding_window(mask_label, percent_overlap=0.4, windowSize=(224, 224)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != 224 or window.shape[1] != 224:
                continue
            if not label_image(window):
                continue
            else:
                label = label_image(window)

            if not os.path.exists(dir2Save+str(label)+"/"):
                os.makedirs(dir2Save+str(label)+"/")

            save_path = dir2Save+ str(label)+"/" + name_im +"_"+institute+"_x_ini_"+str(x)+"_y_ini_"+str(y)+".png"
            io.imsave(save_path, im_RGB[y:y+224,x:x+224])
            df_im = pd.DataFrame({'im':[im],'patch': [ name_im +"_"+institute+"_x_ini_"+str(x)+"_y_ini_"+str(y)+".png"
    ], 'name': [name_im], 'institute': [institute], 'label':[label], 'x':[x], 'y':[y]})
            evaluation_set_patches = [evaluation_set_patches, df_im]
            evaluation_set_patches = pd.concat(evaluation_set_patches)


    evaluation_set_patches.to_csv("corrected_set_patches.csv", float_format="%.0f")
