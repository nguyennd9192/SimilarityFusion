


import numpy as np
from sklearn.preprocessing import LabelEncoder
from general_lib import *
from keras.utils import np_utils


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