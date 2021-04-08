# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 19:35:36 2019

@author: Miguel
"""

import pandas as pd
import skimage.io as io
import numpy as np
import os
# some_file.py
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, "/work/work_mik/CrowdsourcingDataset-Amgadetal2019-master/scripts_database")
sys.path.append("/work/work_mik/CrowdsourcingDataset-Amgadetal2019-master/scripts_database")
print(sys.path)
#sys.path.append('../scripts_database/')
from extracting_patches import labeling, label_image
import tqdm

#dictionaries of the labels
annotator_dict= pd.read_csv("../private-data-Amgadetal2019/annotator_dict.tsv",sep='\t', header=None)

#Core_set patches with labels
core_set = pd.read_csv("../script_database/corrected_set_patches.csv", index_col=0)
#malformed uncorrected masks of the core set
malformed = pd.read_csv("../private-data-Amgadetal2019/malformed_masks_uncorrected_core_set.tsv",sep='\t',index_col=0)

f = lambda x: x["slide_name"].split("-")[2]
names_malformed = set(malformed.apply(f,1).values)

core_set_unmalf = core_set[~core_set.name.isin(eval_malformed)]
core_set_unmalf = core_set_unmalf.reset_index(drop=True)
print("Core unmalformed: ", len(core_set_unmalf))

path_masks='../private-data-Amgadetal2019/uncorrected_core_set/'
uncorrected_mask_list = os.listdir(path_masks)
masks_unc_data = pd.DataFrame({'Annotator': [], 'Patient': [], 'path':[]})

for file in uncorrected_mask_list:
    if file.endswith('.png'):
        file_split=file.replace('-','_').split('_')
        if file_split[1] == 'Abo':
            masks_unc_data2 = pd.DataFrame({'Annotator': [file_split[1]+'-'+file_split[2]], 'Patient': [file_split[6]], 'path': [file]})

        elif file_split[5] == 'TCGA':
            masks_unc_data2 = pd.DataFrame({'Annotator': [file_split[1]], 'Patient': [file_split[4]], 'path': [file]})
        else:
            masks_unc_data2 = pd.DataFrame({'Annotator': [file_split[1]], 'Patient': [file_split[5]], 'path': [file]})

        masks_unc_data = pd.concat([masks_unc_data, masks_unc_data2])

masks_unc_data=masks_unc_data[masks_unc_data["Annotator"]!="Group3"]
patch_list_labels=[]


j=0 #discordance counter
discordance_list=[]
for i,patch in tqdm.tqdm(core_set_unmalf.iterrows()):
    patient = patch['name']
    x,y=patch['x'],patch['y']
    institue = patch["institute"]
    ground_truth=patch['label']
    y_mask = masks_unc_data[masks_unc_data["Patient"]==patient]

    try:
        mask_annotation = io.imread((path_masks+y_mask["path"])[0])[y:y+224,x:x+224]
    except:
        print("We can't open ", patch["patch"])
        patch_list_labels.append(None)
        discordance_list.append(None)
        continue
    patch_annotation = labeling(mask_annotation)
    label = label_image(patch_annotation)
    name_annotator= y_mask["Annotator"][0]
    try:
        code_annotator = annotator_dict[annotator_dict.iloc[:,0]==name_annotator].index.item()
    except ValueError:
        print("No está el anotador en el dict", patch["patch"], name_annotator)
        patch_list_labels.append(None)
        discordance_list.append(None)

        continue
    if not label:
        print("No label, del anotador", patch["patch"])
        patch_list_labels.append(None)
        discordance_list.append(None)
    else:
        if label!=patch['label']:
            j=j+1
            discordance_list.append(True)
        else:
            discordance_list.append(False)
        print("Anotado: ",label, " GT: ", patch['label'], " Discordancias: ", j)
        patch_list_labels.append(np.array([code_annotator,label]))
    print(i, patch["patch"], "Len anotaciones ", len(patch_list_labels), " Len discord: ", len(discordance_list))

print(len(extra_eval_unmalf), len(patch_list_labels))
core_set_unmalf['annotations']=patch_list_labels
core_set_unmalf.to_csv("core_set_patches_annotations.csv")
core_set_unmalf['discordance']=discordance_list
core_set_unmalf.to_csv("core_set_patches_annotations.csv")
core_set_unmalf.to_pickle("core_set_patches_annotations.pkl")
