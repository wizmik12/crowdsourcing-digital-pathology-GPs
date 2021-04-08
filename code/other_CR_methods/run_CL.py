import sys
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense
import os
import time

N_EPOCHS = 20
BATCH_SIZE = 64
SEED = 1
GC = 3

for type in ['MW', 'VW', 'VWB']:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(int(GC))



    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    import CrowdLayer.crowd_layer.crowd_aggregators as crowd_aggregators
    import CrowdLayer.crowd_layer.crowd_layers as crowd_layers
    CrowdsClassification = crowd_layers.CrowdsClassification
    MaskedMultiCrossEntropy = crowd_layers.MaskedMultiCrossEntropy
    CrowdsCategoricalAggregator = crowd_aggregators.CrowdsCategoricalAggregator

    np.random.seed(SEED)
    tf.set_random_seed(SEED)

    ## LOAD and normalize data

    X_train = np.load('../svgpcr_method/features_pretrained/X_train_512pool.npy')
    X_test = np.load('../svgpcr_method/features_pretrained/X_test_512pool.npy')
    Y_train = np.load('./annotations/Y_train_1h.npy')
    y_test = np.load('../svgpcr_method/features_pretrained/y_test.npy')
    y_test_class = np.argmax(y_test,1)

    m,s=X_train.mean(0),X_train.std(0)

    X_train = (X_train - m)/s
    X_test = (X_test - m)/s

    X_train_sub, X_valid, y_train_sub, y_valid = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)


    ## AUXILIARY FUNCTIONS

    N_CLASSES = Y_train.shape[1]
    N_ANNOT = Y_train.shape[2]

    def build_base_model():
        base_model = Sequential()
        base_model.add(Dense(128, activation='relu'))
        base_model.add(Dropout(0.5))
        base_model.add(Dense(N_CLASSES))
        base_model.add(Activation("softmax"))
        base_model.compile(optimizer='adam', loss='categorical_crossentropy')
        return base_model

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

    path = "DL_CL_{0}/seed_{1}/".format(type,SEED)
    assert (not os.path.exists(path)), "The path already exists"
    os.makedirs(path)

    ## TRAINING THE MODEL

    if type=="VWB":
        type2 = "VW+B"
    else:
        type2 = type

    model = build_base_model()
    loss = MaskedMultiCrossEntropy().loss
    cr_layer = CrowdsClassification(N_CLASSES, N_ANNOT, conn_type=type2)
    for n in range(N_EPOCHS):
        model.add(cr_layer)
        model.compile(optimizer='adam', loss=loss)
        start = time.time()
        hist = model.fit(X_train_sub, y_train_sub,
                         epochs=1, shuffle=True, batch_size=BATCH_SIZE, verbose=2)
        trainTime = time.time()-start
        weights = model.layers[4].get_weights()
        model.pop()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        probs = model.predict(X_test)
        preds = np.argmax(probs,1)
        metrics, mat, acc, NLL=eval_performance(preds, y_test_class, probs, output=None)
        print('epoch', str(n), ': \n', metrics)

    probs = model.predict(X_test)
    preds = np.argmax(probs,1)

    metrics, mat, acc, NLL = eval_performance(preds, y_test_class, probs, output=None)
    roc_macro = roc_auc_score(y_test, probs, average='macro')
    print('AUC', roc_macro)

    with open('./Results/results_CL' + type + '.txt', 'w') as the_file:
        the_file.write(metrics)
        the_file.write('\n\nAUC  ' + str(roc_macro) +'\nNLL ' + str(NLL))
