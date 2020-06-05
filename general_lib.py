

input_dir = "/Users/nguyennguyenduong/Dropbox/My_code/SimilarityFusion/input"
result_dir = "/Users/nguyennguyenduong/Dropbox/My_code/SimilarityFusion/result"


import yaml, sys, gc, os, ntpath

import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, precision_recall_fscore_support

def release_mem(fig):
    fig.clf()
    plt.close()
    gc.collect()


def get_basename(filename):
    import ntpath

    head, tail = ntpath.split(filename)

    basename = os.path.splitext(tail)[0]
    return tail

def makedirs(file):
    if not os.path.isdir(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))



def get_Xnorm(X_matrix):

    min_max_scaler = preprocessing.MinMaxScaler()
    x_normed = min_max_scaler.fit_transform(X_matrix)
    #x_normed = X_matrix
    return x_normed



def normalize(X_train, X_test=None):

    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_norm = min_max_scaler.fit_transform(X_train)

    if X_test is not None:
        X_test_norm = min_max_scaler.fit_transform(X_test)
        return X_train_norm, X_test_norm
    else:
        return X_train_norm


def modelEvaluation(estimator, X_test, y_test):
    y_pred = estimator.predict(X_test)
    r2 = r2_score(y_pred, y_test)
    mae = mean_absolute_error(y_pred, y_test)
    rmse = np.sqrt(mean_squared_error(y_pred, y_test))

    print( "R2:", r2)
    print( "MAE:", mae)
    print( "RMSE:", rmse)
    return r2, mae, rmse


def modelEvaluationClf(estimator, X_test, y_test):
    y_pred = estimator.predict(X_test)
    this_score = precision_recall_fscore_support(y_true=y_test, y_pred=y_pred, 
                average='macro')

    prec, recall, f1, support = this_score
    return prec, recall, f1, support