# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 02:42:29 2019

@author: Miguel

Vamos a extraer las etiquetas por MV en el evaluation set pero haciendo
MV por pixel.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import skimage.io as io
from extracting_patches import labeling, label_image
from tqdm import tqdm
from scipy import stats

#Return the name of the annotator of the mask
def name_annotator_fun(file):
    file_split=file.replace('-','_').split('_')
    if file_split[0] == 'Abo':
        return file_split[0]+'-'+file_split[1]
    elif file_split[0] == 'Habiba.EL':
        return file_split[0]+'-'+file_split[1]
    else:
        return file_split[0]

#dictionaries of the labels
annotator_dict= pd.read_csv("../private-data-Amgadetal2019/annotator_dict.tsv",sep='\t', header=None) #Expertise of each annotator dict
malformed = pd.read_csv("../private-data-Amgadetal2019/malformed_masks_evaluation.tsv",sep='\t',index_col=0) #List of malformed mask that we have to remove
mask_path = '../private-data-Amgadetal2019/evaluation_set_10_slides/' #Path of the annotation masks of the evaluation set

#Removing bad masks
mask_directory = os.listdir(mask_path)
list_mal = []
for x in mask_directory:
    if 'vgg' in x:
        list_mal.append(x)
        continue
    for y in range(len(malformed)):
        y_malformed=malformed.iloc[y,:]
        if (y_malformed['Participant'] in x) and (y_malformed['slide_name'] in x):
            list_mal.append(x)
mask_directory=list(set(mask_directory)-set(list_mal)) #List of good masks to take into account in the experiments

"""
masks_eval_data = pd.DataFrame({'Annotator': [], 'Patient': [], 'path':[]})
for file in mask_directory:
    if file.endswith('.png'):
        file_split=file.replace('-','_').split('_')
        if file_split[0] == 'Abo':
            masks_eval_data2 = pd.DataFrame({'Annotator': [file_split[0]+'-'+file_split[1]], 'Patient': [file_split[5]], 'path':[file]})
        elif file_split[0] == 'Habiba.EL':
            masks_eval_data2 = pd.DataFrame({'Annotator': [file_split[0]+'-'+file_split[1]], 'Patient': [file_split[5]], 'path':[file]})
        else:
            masks_eval_data2 = pd.DataFrame({'Annotator': [file_split[0]], 'Patient': [file_split[4]], 'path':[file]})
        masks_eval_data = pd.concat([masks_eval_data, masks_eval_data2])
"""

patches_df = pd.read_csv("evaluation_set_patches.csv", index_col=0, dtype={"x": int, "y": int})

list_labels=[]
for i,patch in tqdm(patches_df.iterrows()):
    patient = patch['name']
    x,y=patch['x'],patch['y']
    ground_truth=patch['label']
    mask_annotation_list=[]
    for y_mask in mask_directory:
        if patient in y_mask:
            try:
                name_annotator=name_annotator_fun(y_mask)
                code_annotator = annotator_dict[annotator_dict.iloc[:,0]==name_annotator].index.item()
            except ValueError:
                continue
            if code_annotator<=4:
                continue
            mask_annotation_list.append(io.imread(mask_path+y_mask)[y:y+224,x:x+224])

    mask_annotation_list=np.array(mask_annotation_list)
    mask_annotation=np.squeeze(stats.mode(mask_annotation_list,axis=0)[0])
    print('sizes: ', mask_annotation_list.shape, mask_annotation.shape)
    patch_annotation = labeling(mask_annotation)
    label = label_image(patch_annotation)
    print(label)
    list_labels.append(label)

patches_df['annotations']=list_labels
patches_df.to_csv("evaluation_set_patches_MV.csv", float_format="%.0f")
patches_df.to_pickle("evaluation_set_patches_MV.pkl")
