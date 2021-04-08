import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import skimage.io as io
import pandas as pd



def labeling(annotation_mask):
    #Convert the annotation mask into a label mask (the three that we are considering tumor, stroma and other).
    label_mask = -1*np.ones(annotation_mask.shape)
    label_mask[np.logical_or(annotation_mask==1,annotation_mask==19,annotation_mask==20)] = 1 #tumor
    label_mask[annotation_mask==2] = 2 #stroma
    label_mask[np.logical_or(annotation_mask==3,annotation_mask==10,annotation_mask==11)] = 3 #inflammation
    return label_mask

def label_image(label_mask):
    #If the mask has more than the 50% of one class assigns it to this class.
    pixels = label_mask.shape[0]*label_mask.shape[1]
    for i in range(1,4):
        if (label_mask==i).sum()/pixels >= 0.5:
            return i
    return False

def sliding_window(image, percent_overlap, windowSize):
    # slide a window across the image
    stepSize = int((1 - percent_overlap) * windowSize[0])
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (int(x), int(y), image[y:y + windowSize[1], x:x + windowSize[0]])

if __name__ == "__main__":

    #This is only for evaluation_set
    dir2Save = "../patches_evaluation/"
    if not os.path.exists(dir2Save):
        os.makedirs(dir2Save)

    #We have two directories to extract patches, we choose between them.
    images_path = '../eval_set_normalized/'
    #images_path = './evaluation_set_RGB/'
    images_directory = os.listdir(images_path)
    mask_path = '../eval_masks_to_train_model/SP/'
    #mask_path = './eval_masks_to_train_model/SP/'
    mask_directory = os.listdir(mask_path)


    evaluation_set_patches = pd.DataFrame({'im':[],'patch':[],'name': [], 'institute': [], 'label':[], 'x':[], 'y':[]})
    for im in images_directory:
        print("Reading image ", im)
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


    evaluation_set_patches.to_csv("evaluation_set_patches.csv", float_format="%.0f")
