import time
import pickle

import scipy.io as sio
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix,f1_score, precision_score, recall_score, log_loss, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import gpflow
from gpflow.models import SVGP


def batch_prediction_multi_classification(model, prediction_model, X, S):
    n_batches = max(int(X.shape[0]/50.), 1)
    pred_list = []
    prob_list = []
    for X_batch in zip(np.array_split(X, n_batches)):
        X_batch = np.array(X_batch)
        X_batch =  np.squeeze(X_batch, axis=0)
        pred, prob = prediction_model(model, X_batch, S)
        pred_list.append(pred)
        prob_list.append(prob)
    pred = np.concatenate(pred_list, 0)
    prob = np.concatenate(prob_list, 0)
    return pred, prob

def prediction_multi_dgp(model, X_batch, S):
    m, v = model.predict_y(X_batch, S)
    prob = np.average(m,0)
    return np.argmax(prob,1), prob

def plot_confusion_matrix(cm, title, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    classes = ["NOISE", "EXP", "REG", "COL", "VTE", "TRE", "LPE"]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    image =   title + ".png"
    fig.savefig(image)   # save the figure to file
    plt.close(fig)
    return ax

def eval_performance(pred, label_test, prob, output=None):
    cm = confusion_matrix(label_test, pred)
    mat = cm.astype('float')# / cm.sum(axis=1)[:, np.newaxis]
    if output == 'df':
        metrics = classification_report(label_test, pred, output_dict=True)
        metrics = pd.DataFrame(metrics)
    else:
        metrics = classification_report(label_test, pred, digits=4)
    acc = accuracy_score(label_test, pred)
    NLL = log_loss(label_test, prob)

    return metrics, mat, acc, NLL

def get_metrics(X_valid, y_valid, m):
    try:
        preds_valid, prob_valid = batch_prediction_multi_classification(m, prediction_multi_dgp, X_valid, S=100)
    except:
        prob_valid = m.predict_y(X_valid)[0]
        preds_valid = np.argmax(prob_valid, 1).astype(int)

    return eval_performance(preds_valid, y_valid.astype(int), prob_valid)

def train_GP_model(m, X_train, X_valid, y_valid, num_inducing):
    #Iters
    iters = [1000, 20000, 10000]
    trainTime_e = []
    ELBO = []
    iter_notif = 100
    #Sesion
    sess = m.enquire_session()
    optop = gpflow.train.AdamOptimizer(0.01).make_optimize_tensor(m)
    optop2 = gpflow.train.AdamOptimizer(0.001).make_optimize_tensor(m)

    try:
        for layer in m.layers[:-1]:
            layer.feature.Z.set_trainable(False)
    except:
        m.feature.Z.set_trainable(False)

    print("0 training starts:")
    start = time.time()
    for _ in range(iters[0]):
        sess.run((m.likelihood_tensor, optop))
    print(m.as_pandas_table())
    trainTime_e.append(time.time()-start)
    print("Finished:", trainTime_e[-1])
    m.anchor(sess)
    ELBO.append(np.mean(np.array([m.compute_log_likelihood() for a in range(10)])))
    print('======================training 0 : =======================')
    print('ELBO: ', ELBO[-1],'\n')
    metrics = get_metrics(X_valid, y_valid, m)
    print(metrics[0])

    if X_train.shape[0] > num_inducing:
        try:
            for layer in m.layers[:-1]:
                layer.feature.Z.set_trainable(True)
        except:
            m.feature.Z.set_trainable(True)

    print("1 training starts:")
    start = time.time()
    for j in range(int(iters[1]/iter_notif)):
        for _ in range(iter_notif):
            sess.run((m.likelihood_tensor, optop))
        m.anchor(sess)
        ELBO.append(np.mean(np.array([m.compute_log_likelihood() for a in range(10)])))
        print('======================iter, ',j, ' of ', int(iters[1]/iter_notif), ': =======================')
        print('ELBO: ', ELBO[-1],'\n')
        metrics = get_metrics(X_valid, y_valid, m)
        print(metrics[0])



    trainTime_e.append(time.time()-start)
    print("Finished train 2:", trainTime_e[-1])

    print("2 training starts:")
    start = time.time()
    for j in range(int(iters[2]/iter_notif)):
        for _ in range(iter_notif):
            sess.run((m.likelihood_tensor, optop2))
        m.anchor(sess)
        ELBO.append(np.mean(np.array([m.compute_log_likelihood() for a in range(10)])))
        print('======================iter, ',j, ' of ', int(iters[2]/iter_notif), ': =======================')
        print('ELBO: ', ELBO[-1],'\n')
        metrics = get_metrics(X_valid, y_valid, m)
        print(metrics[0])

    trainTime_e.append(time.time()-start)
    print("Finished train 2:", trainTime_e[-1])
    m.anchor(sess)
    return m
