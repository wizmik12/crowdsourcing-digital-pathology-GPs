import os
import pandas as pd
import shutil
import numpy as np
#dictionaries of the labels
annotator_dict= pd.read_csv("../private-data-Amgadetal2019/annotator_dict.tsv",sep='\t', header=None)

for i in range(1,4):
    if not os.path.exists('./Train_non_experts_simple/' + str(i) + '/'):
        os.makedirs('./Train_non_experts_simple/' + str(i) + '/')
new_root_train = './Train_non_experts_simple/'

core_set = pd.read_pickle("core_set_patches_annotations.pkl")
eval_set=pd.read_pickle("../scripts_database/evaluation_set_patches_annotations.pkl")

len_eval_set=len(eval_set)
i=0
annotations_non_experts = pd.DataFrame({'patch':[], 'annotations':[], 'discordance':[], 'label':[]})
for root, subdirs, files in os.walk('../patches_evaluation/'):
    for name in files:
      i=i+1
      print(name, i,"/", len_eval_set)
      annotations = eval_set[eval_set["patch"]==name]["annotations"].values[0]
      if np.any(annotations == None):
          print("None ",name)
          continue
      if np.any(annotations[:,0]>4):
          rename = eval_set[eval_set["patch"]==name]["im"].values[0].split('.')[0] + name
          label = eval_set[eval_set["patch"]==name]["label"].values[0]
          annotations=annotations[annotations[:,0]>4]

          new_root_label = os.path.join(new_root_train,  str(label))
          shutil.copyfile(os.path.join(root,name), os.path.join(new_root_label,rename))
          annotations_non_experts=annotations_non_experts.append({'patch':rename, 'annotations': annotations, "discordance": None, "label": label}, ignore_index=True)
      else:
          print("Only experts, ", name)

i=0
len_core_set=len(core_set)
for root, subdirs, files in os.walk('../patches_core/'):
    for name in files:
      i=i+1
      print(name, i,"/", len_core_set)
      try:
          annotations = core_set[core_set["patch"]==name]["annotations"].values[0].reshape(-1,2)
      except AttributeError:
          print("Seems like None ", name)
          continue
      except IndexError:
          print("Seems like this patch is not annotated ", name)
          continue
      if np.any(annotations == None):
          print("None ",name)
          continue
      if np.any(annotations[:,0]>4):
          rename = core_set[core_set["patch"]==name]["im"].values[0].split('.')[0] + name
          label = core_set[core_set["patch"]==name]["label"].values[0]
          annotations=annotations[annotations[:,0]>4]
          disc = core_set[core_set["patch"]==name]["discordance"].values[0]

          new_root_label = os.path.join(new_root_train,  str(label))
          shutil.copyfile(os.path.join(root,name), os.path.join(new_root_label,rename))
          annotations_non_experts=annotations_non_experts.append({'patch':rename, 'annotations': annotations, "discordance": disc, "label": label}, ignore_index=True)
      else:
          print("Only experts, ", name)
annotations_non_experts.to_pickle('annotations_non_experts_simple.pkl')
