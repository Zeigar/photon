#!/usr/bin/env python3

# Add project_root to path s.t. we can run this script from terminal
import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../..'))

from Framework.PhotonBase import Hyperpipe, PipelineElement
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

mnist = fetch_mldata('MNIST original')
train_data, test_data, train_labels, test_labels = train_test_split(
    mnist.data/255.,
    mnist.target,
    test_size=1/7
)

cv = KFold(n_splits=3, shuffle=True, random_state=0)

n_train_data = len(train_data)
pipe = Hyperpipe(
    'mnistsvm_gridsearch',
    cv,
    optimizer='timeboxed_random_grid_search',
    optimizer_params={
        'log': {'path': 'logs/', 'name': 'mnistsvm_gridsearch'}
    },
    metrics=['accuracy'],
    verbose=2,
    logging=True
)
param_range = np.array(range(int(np.exp(-10)*10), int(np.exp(10)*10)))/10
pipe += PipelineElement.create('svc', {'C': param_range, 'gamma': param_range})

pipe.fit(train_data, train_labels)
