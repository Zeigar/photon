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

os.environ["CUDA_VISIBLE_DEVICES"]="1"

db = photonDB()
data = db.data
data = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2], data.shape[3], 1))
gender = db.gender
n_data = gender.shape[0]

le = LabelEncoder()
gender = le.fit_transform(gender)

# test the complete hyperparameter search with KFold(n_splits=3)
my_pipe = Hyperpipe(
    'CNN3D', ShuffleSplit(n_splits=3), metrics=['accuracy'],
    optimizer='fabolas', optimizer_params={
        'n_min_train_data': int(n_data/9000), 'n_train_data': n_data
    },
    eval_final_performance=True)

my_pipe += PipelineElement.create('CNN3d', {'conv1Filters': [32, 64, int],
                                           'momentum': [0.8, 0.95, float],
                                           'learning_rate': [0.0005, 0.005, float],
                                           'learningRateDecay':[0.0001, 0.001, float],
                                            'nb_epochs': [40, 80, int]},
                                  target_dimension=2, gpu_device='/gpu:2', metrics=['accuracy'])


my_pipe.fit(data, gender)
