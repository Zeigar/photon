#!/usr/bin/env python3

# Add project_root to path s.t. we can run this script from terminal
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../..'))

from sklearn.preprocessing import LabelEncoder
from Framework.PhotonBase import Hyperpipe, PipelineElement
import numpy as np
from photonDB import photonDB
from sklearn.model_selection import ShuffleSplit
from keras_utils import get_rois

os.environ["CUDA_VISIBLE_DEVICES"]="1"

db = photonDB()
data = db.data

#test for small trainingset
masks = get_rois(atlas='aal', rois=[])
data_masked = data * np.tile(masks['whole_brain'], (data.shape[0], 1, 1, 1))

data_vec = np.zeros((int(data_masked.shape[0]),int(np.sum(masks['whole_brain']))))

for i in range(data_masked.shape[0]):
    data_vec[i,:] = data_masked[i,:,:,:][data_masked[i,:,:,:]!=0]
data = data_vec




data = np.reshape(data, (data.shape[0], data.shape[1], 1))
gender = db.gender

le = LabelEncoder()
gender = le.fit_transform(gender)

n_data = int(len(gender)/9)

# test the complete hyperparameter search with KFold(n_splits=3)
my_pipe = Hyperpipe(
    'CCNN', ShuffleSplit(n_splits=3), metrics=['accuracy'],
    optimizer='fabolas', optimizer_params={
    'n_min_train_data': int(n_data/9000), 'n_train_data': n_data},
    eval_final_performance=True)

my_pipe += PipelineElement.create('CCNN', {'conv1Filters': [32, 64, int],
                                           'momentum': [0.8, 0.95, float],
                                           'learning_rate': [0.0005, 0.005, float],
                                           'learningRateDecay':[0, 0.01, float],
                                           'conv1KernelSize':[5, 15, int],
                                           'conv1KernelStrides':[1, 15, int],
                                           'gausNoiseStddev': [0, 1.0, float]},
                                  target_dimension=2, gpu_device='/gpu:0', nb_epochs=80, metrics=['accuracy'])

my_pipe.fit(data, gender)
