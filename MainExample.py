# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 16:36:16 2020

@author: Programmer
"""
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib 
from keras.layers.core import Dropout, Activation
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

import os, ntpath
import pandas as pd
input_dir = "/Users/nguyennguyenduong/Dropbox/My_code/SimilarityFusion/NDD"
result_dir = "/Users/nguyennguyenduong/Dropbox/My_code/SimilarityFusion/result"


def makedirs(file):
    if not os.path.isdir(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))

def get_basename(filename):

    head, tail = ntpath.split(filename)

    basename = os.path.splitext(tail)[0]
    return tail
#--------------------------------------------------
#NDD Methods
def prepare_data(sim_file, seperate=False):

    drug_fea = np.loadtxt(sim_file, dtype=float, delimiter=",")
    interaction = np.loadtxt("{0}/DS1/drug_drug_matrix.csv".format(input_dir),dtype=int,delimiter=",") # drug_drug_matrix
    train = []
    label = [] # # label means drugX interact with drugY or not, stored only in "drug_drug_matrix"
    tmp_fea=[]
    drug_fea_tmp = []

    # # to produce X_data1, X_data2 in training, testing
    for i in range(0, interaction.shape[0]):
        for j in range(0, interaction.shape[1]):
            label.append(interaction[i,j])
            drug_fea_tmp = list(drug_fea[i])
            if seperate:
        
                 tmp_fea = (drug_fea_tmp, drug_fea_tmp)

            else:
                 tmp_fea = drug_fea_tmp + drug_fea_tmp
            train.append(tmp_fea)

    return np.array(train), label
#--------------------------------------------------------------
def calculate_performace(test_num, pred_y,  labels):
    tp =0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] ==1:
            if labels[index] == pred_y[index]:
                tp = tp +1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn +1
            else:
                fp = fp + 1 
    acc = float(tp + tn)/test_num
    if tp == 0 and fp == 0:
        precision = 0
        MCC = 0
        sensitivity = float(tp)/ (tp+fn)
        specificity = float(tn)/(tn + fp)
    else:
        precision = float(tp)/(tp+ fp)
        sensitivity = float(tp)/ (tp+fn)
        specificity = float(tn)/(tn + fp)

        try:
            MCC = float(tp*tn-fp*fn)/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
        except Exception as e:
            MCC = None
            pass
    return acc, precision, sensitivity, specificity, MCC 
#-----------------------------------------------------
def transfer_array_format(data):
    formated_matrix1 = []
    formated_matrix2 = []
    for val in data:
        formated_matrix1.append(val[0])
        formated_matrix2.append(val[1])
    return np.array(formated_matrix1), np.array(formated_matrix2)
#-------------------------------------------------------
def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
        y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
        print(y)
    return y, encoder
#------------------------------------------------------
def preprocess_names(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    if categorical:
        labels = np_utils.to_categorical(labels)
    return labels, encoder
#------------------------------------------------------
def NDD(input_dim): 
    model = Sequential()
    model.add(Dense(input_dim=input_dim, output_dim=400,init='glorot_normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(input_dim=400, output_dim=300,init='glorot_normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(input_dim=300, output_dim=2,init='glorot_normal'))
    model.add(Activation('sigmoid'))
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd)                  
    return model
#--------------------------------------------------

#---------------------------------------------------------------------------------------------------
def DeepMDA(sim_file):

    X, labels = prepare_data(sim_file=sim_file, seperate=True)
    X_data1, X_data2 = transfer_array_format(X) 
    X=0
    y, encoder = preprocess_labels(labels)# labels labels_new

    X= np.concatenate((X_data1, X_data2), axis = 1)
    num = np.arange(len(y))
    np.random.shuffle(num)
    X_data1 = X_data1[num]
    X_data2 = X_data2[num]
    y = y[num]
    num_cross_val = 10

   
    clf_results = []

    for fold in range(num_cross_val):
        train_label = np.array([x for i, x in enumerate(y) if i % num_cross_val != fold])
        test_label = np.array([x for i, x in enumerate(y) if i % num_cross_val == fold])
        train1 = np.array([x for i, x in enumerate(X_data1) if i % num_cross_val != fold])
        test1 = np.array([x for i, x in enumerate(X_data1) if i % num_cross_val == fold])
        train2 = np.array([x for i, x in enumerate(X_data2) if i % num_cross_val != fold])
        test2 = np.array([x for i, x in enumerate(X_data2) if i % num_cross_val == fold])
     
        zerotest=0
        nozerotest=0
        zerotrain=0
        nozerotrain=0
        real_labels = []
        for val in test_label:
            if val[0] == 1:
                nozerotest=nozerotest+1
                real_labels.append(0)
            else:
                zerotest=zerotest+1
                real_labels.append(1)
        train_label_new = []
        for val in train_label:
            if val[0] == 1:
                zerotrain=zerotrain+1
                train_label_new.append(0)
            else:
                nozerotrain=nozerotrain+1
                train_label_new.append(1)
       
        prefilter_train = np.concatenate((train1, train2), axis = 1)
        prefilter_test = np.concatenate((test1, test2), axis = 1)
        

        n_train = int(prefilter_train.shape[0])
        # prefilter_train = prefilter_train[:n_train, ]
        print ("train dimensions:", prefilter_train.shape)
        print ("test dimensions:", prefilter_test.shape)

        model_DNN = NDD(input_dim=prefilter_train.shape[1])

        # # chỉ để label ở cột thứ 2 x[:, 1]
        train_label_new_forDNN = np.array([[0,1] if i == 1 else [1,0] for i in train_label_new])

        model_DNN.fit(prefilter_train[:n_train, ],train_label_new_forDNN[:n_train, ],
            batch_size=100,epochs=20,shuffle=True,validation_split=0)
        proba = model_DNN.predict_classes(prefilter_test,batch_size=200,verbose=True)
        ae_y_pred_prob = model_DNN.predict_proba(prefilter_test,batch_size=200,verbose=True)

        print (proba)
        print (ae_y_pred_prob)

        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), proba,  real_labels)
        fpr, tpr, auc_thresholds = roc_curve(real_labels, ae_y_pred_prob[:,1])
        auc_score = auc(fpr, tpr)

        precision1, recall, pr_threshods = precision_recall_curve(real_labels, ae_y_pred_prob[:,1])
        aupr_score = auc(recall, precision1)
        #f = f1_score(real_labels, transfer_label_from_prob(ae_y_pred_prob[:,1]))
        all_F_measure=np.zeros(len(pr_threshods))
        clf_results.append(dict({"sim_file":sim_file, "folds":"fold_{0}".format(fold), 
            "acc":acc, "precision":precision, 
            "sensitivity":sensitivity, "specificity":specificity, "MCC":MCC}))

        df = pd.DataFrame(clf_results)
        saveat = "{0}/{1}_results.csv".format(result_dir, get_basename(sim_file).replace(".csv", ""))
        makedirs(saveat)
        df.to_csv(saveat)
    print ("Results: ", clf_results)











