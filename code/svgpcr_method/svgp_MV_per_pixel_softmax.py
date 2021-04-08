import os
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
import gpflow
import tensorflow as tf
from scipy.cluster.vq import kmeans
import pickle

from utils_svgp import train_GP_model, eval_performance

np.random.seed(0)
tf.set_random_seed(1111)

#Results_folder
if not os.path.exists('./Results/'):
    os.makedirs('./Results/')
#Loading data
X_train, y_train = np.load("./features_pretrained/features_MV_per_pixel.npy").astype('float64'), np.load("./features_pretrained/y_train_MV_per_pixel.npy").astype('float64')
X_test, y_test = np.load("./features_pretrained/X_test_512pool.npy").astype('float64'), np.load("./features_pretrained/y_test.npy").astype('float64')

print('size', X_train.shape, y_train.shape)
m,s=X_train.mean(0),X_train.std(0)
X_train = (X_train - m)/s
X_test = (X_test - m)/s
y_train, y_test = np.argmax(y_train, 1), np.argmax(y_test, 1)
X_train_sub, X_valid, y_train_sub, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print(X_train_sub.shape, y_train_sub.shape)
from collections import Counter
print(Counter(y_train_sub))
num_inducing=100
Z = kmeans(X_train_sub, num_inducing)[0]
minibatch_size = 1000 if X_train_sub.shape[0]>1000 else None

m = gpflow.models.SVGP(X=X_train_sub, Y=y_train_sub.reshape(-1,1), Z=Z, kern=gpflow.kernels.RBF(X_train_sub.shape[1], variance=2.0, lengthscales =2.0, ARD=False), likelihood=gpflow.likelihoods.SoftMax(3), num_latent=3, minibatch_size=minibatch_size)
trained_m = train_GP_model(m, X_train_sub, X_valid, y_valid, num_inducing)

probs = trained_m.predict_y(X_test)[0]
preds = np.argmax(probs,axis=1)

y_test=np.load("./features_pretrained/y_test.npy").astype('float64')
y_test_class = np.argmax(y_test,axis=1)

np.save("probs_SVGP_MV_per_pixel", probs)
metrics, mat, acc, NLL = eval_performance(preds, y_test_class, probs, output=None)
print(metrics, '\n NLL ', NLL)
roc_macro = roc_auc_score(y_test, probs, average='macro')
print('AUC', roc_macro)

with open('./Results/results_svgp_MV_per_pixel_softmax.txt', 'w') as the_file:
    the_file.write(metrics)
    the_file.write('\n\nAUC  ' + str(roc_macro) +'\nNLL ' + str(NLL))


#saver = gpflow.saver.Saver()
#saver.save("svgp_MV.gpflow", m)

dir2Save = './models/'
if not os.path.exists(dir2Save):
    os.makedirs(dir2Save)
path = dir2Save + 'model_svgp_MV_per_pixel_softmax.pkl'
with open(path, 'wb') as fp:
    pickle.dump(trained_m.read_trainables(), fp)
